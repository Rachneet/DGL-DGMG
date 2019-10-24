def dataset_based_configure(opts):

    if opts['dataset'] == 'cycles':
        ds_configure = cycles_configure
    elif opts['dataset'] == 'barabasi':
            ds_configure = barabasi_configure
    elif opts['dataset'] == 'community':
            ds_configure = community_configure
    else:
        raise ValueError('Unsupported dataset: {}'.format(opts['dataset']))

    opts = {**opts, **ds_configure}

    return opts


synthetic_dataset_configure = {
    'node_hidden_size': 16,
    'num_propagation_rounds': 2,
    'optimizer': 'Adam',
    'nepochs': 25,
    'ds_size': 4000,
    'num_generated_samples': 10000,
}

cycles_configure = {
    **synthetic_dataset_configure,
    **{
        'min_size': 10,
        'max_size': 20,
        'lr': 5e-4,
    }
}

barabasi_configure = {
    **synthetic_dataset_configure,
    **{
        'min_size': 4,
        'max_size': 20,
        'lr': 5e-4,
    }
}


community_configure = {
    **synthetic_dataset_configure,
    **{
        'min_size': 28,
        'max_size': 28,
        'lr': 5e-4,
    }
}