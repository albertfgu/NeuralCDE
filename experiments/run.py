import uea

# group = 2 corresponds to 50% mising rate
uea.run_all(
    # train_missing_rate=0.0, eval_missing_rate=0.,
    # train_missing_rate=0.5, eval_missing_rate=0.,
    train_missing_rate=0.0, eval_missing_rate=0.5,
    # train_missing_rate=0.5, eval_missing_rate=0.5,
    train_timescale=1, eval_timescale=1,
    missing_uniform=True, missing_unknown=False,
    device='cuda', dataset_name='CharacterTrajectories',
    model_names=(
        'ncde',
        'gruode',
        'odernn',
        # 'decay',
        # 'dt',
    ),
    # max_epochs=100,
    max_epochs=100,
)
