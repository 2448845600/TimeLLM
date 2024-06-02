model_conf = dict(
    model_name='TimeLLM',
    dataset_name='ETTh1',
    exp_type="long_time_series_forecasting",

    hist_len=512,

    d_model=32,
    d_ff=128,
    n_heads=8,
    patch_len=16,
    stride=8,
    dropout=0.1,
    local_hf_cache_dir="/data0/team_data/hf_models/",
    llm_model='GPT-2',
    llm_dim=768,
    llm_layers=12,

    batch_size=8,
    lr=0.0001,
)