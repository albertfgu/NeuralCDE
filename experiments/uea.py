import common
import datasets
import pprint


def main(
    dataset_name, train_missing_rate=0., eval_missing_rate=0.,                                          # dataset parameters
    missing_uniform=True, missing_unknown=True,
    train_timescale=1, eval_timescale=1,
    device='cuda', max_epochs=1000, *,                                       # training parameters
    model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,  # model parameters
    dry_run=False,
    **kwargs):                                                               # kwargs passed on to cdeint

    params = {
        'train hz': 1.-train_missing_rate,
        'eval hz': 1.-eval_missing_rate,
        'uniform': missing_uniform,
        'timestamp': not missing_unknown,
        'train ts': train_timescale,
        'eval ts': eval_timescale,
    }
    print(f"Experiment: Model {model_name}")
    pprint.pprint(params)
    # print("\n    train missing rate {train_missing_rate}\n    eval missing rate {eval_missing_rate}\n    train timescale {train_timescale}\n    uniform spa")

    batch_size = 32
    lr = 0.001 * (batch_size / 32)

    # Need the intensity data to know how long to evolve for in between observations, but the model doesn't otherwise
    # use it because of use_intensity=False below.
    intensity_data = True if model_name in ('odernn', 'dt', 'decay') else False
    # intensity_data = False

    (times, train_dataloader, val_dataloader,
     test_dataloader, num_classes, input_channels) = datasets.uea.get_data_shift(
        dataset_name,
        train_missing_rate, eval_missing_rate,
        missing_uniform, missing_unknown,
        train_timescale, eval_timescale,
        intensity=intensity_data,
        device=device,
        batch_size=batch_size,
    )

    if num_classes == 2:
        output_channels = 1
    else:
        output_channels = num_classes

    make_model = common.make_model(model_name, input_channels, output_channels, hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers, use_intensity=False, initial=True)

    if dry_run:
        name = None
    else:
        name = dataset_name + str(int(train_missing_rate * 100)) + '-' + str(int(eval_missing_rate * 100))
    return common.main(name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model,
                       num_classes, max_epochs, lr, kwargs, step_mode=False)
                       # num_classes, max_epochs, lr, kwargs, step_mode=True)


def run_all(
        dataset_name,
        train_missing_rate=0., eval_missing_rate=0.,
        missing_uniform=False, missing_unknown=False,
        train_timescale=1, eval_timescale=1,
        device='cuda', max_epochs=100, model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode'), trials=1):

    model_kwargs = dict(ncde=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
                        odernn=dict(hidden_channels=32, hidden_hidden_channels=32, num_hidden_layers=3),
                        dt=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=47, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(trials):
            results = main(
                    dataset_name,
                    train_missing_rate, eval_missing_rate,
                    missing_uniform, missing_unknown,
                    train_timescale, eval_timescale,
                    device,max_epochs=max_epochs,
                    model_name=model_name, **model_kwargs[model_name])
            main_results = {
                'train loss': results.train_metrics.loss,
                'train acc': results.train_metrics.accuracy,
                'val loss': results.val_metrics.loss,
                'val acc': results.val_metrics.accuracy,
                'test loss': results.test_metrics.loss,
                'test acc': results.test_metrics.accuracy,
            }
            pprint.pprint(main_results)
