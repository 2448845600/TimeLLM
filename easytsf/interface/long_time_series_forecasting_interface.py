import importlib
import inspect
import os

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs


class ExpInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        stat = np.load(os.path.join(self.hparams.data_root, self.hparams.dataset_name, 'var_scaler_info.npz'))
        self.register_buffer('mean', torch.tensor(stat['mean']).float())
        self.register_buffer('std', torch.tensor(stat['std']).float())

        self.test_result = []

    def forward(self, batch, batch_idx):
        norm_x = batch[0].float()
        norm_y = batch[1].float()[:, -self.hparams.pred_len:, :, 0]
        raw_y = self.inverse_transform(norm_y)

        norm_output = self.model(norm_x)[:, -self.hparams.pred_len:, :]
        raw_output = self.inverse_transform(norm_output)
        return raw_output, raw_y, norm_output, norm_y

    def training_step(self, batch, batch_idx):
        _, _, norm_output, norm_y = self.forward(batch, batch_idx)
        loss = self.loss_function(norm_output, norm_y)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, norm_output, norm_y = self.forward(batch, batch_idx)
        loss = self.loss_function(norm_output, norm_y)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, _, norm_output, norm_y = self.forward(batch, batch_idx)
        mae = torch.nn.functional.l1_loss(norm_output, norm_y)
        mse = torch.nn.functional.mse_loss(norm_output, norm_y)
        self.log('test/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.optimizer_weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, betas=self.hparams.optimizer_betas,
                weight_decay=self.hparams.optimizer_weight_decay)
        else:
            raise ValueError('Invalid optimizer type!')

        if self.hparams.lr_scheduler == 'StepLR':
            lr_scheduler = {
                "scheduler": lrs.StepLR(
                    optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
            }
        elif self.hparams.lr_scheduler == 'MultiStepLR':
            lr_scheduler = {
                "scheduler": lrs.MultiStepLR(
                    optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
            }
        elif self.hparams.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = {
                "scheduler": lrs.ReduceLROnPlateau(
                    optimizer, mode='min', factor=self.hparams.lrs_factor, patience=self.hparams.lrs_patience),
                "monitor": self.hparams.val_metric
            }
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def configure_loss(self):
        self.loss_function = nn.MSELoss()

    def load_model(self):
        model_name = self.hparams.model_name
        try:
            Model = getattr(importlib.import_module('.' + model_name.lower(), package='easytsf.model'), model_name)
        except:
            raise ValueError(f'Invalid Module File Name {model_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        model_class_args = inspect.getfullargspec(Model.__init__).args[1:]  # 获取模型参数
        interface_args = self.hparams.keys()
        model_args_instance = {}
        for arg in model_class_args:
            if arg in interface_args:
                model_args_instance[arg] = getattr(self.hparams, arg)
        return Model(**model_args_instance)

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
