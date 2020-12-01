RNN_CONFIG = {
    'num_layers': 2,
    'hidden_size': 128,
    'random_input': 100,
    'dropout': 0.3,
    'music_size': 89  # 88 notes + silence
}

TRAIN_CONFIG = {
    'batch_size': 128,
    'workers': 6,
    'K': 1,
    'epochs': 100,
    'learning_rate': 3e-2,
    'weight_decay': 0,
    'experiment_name': 'short_v2',
    'balance': True,
    'feature_matching': False,
    'save_every_n_epochs': 5,
    'load_G': None,  # 'checkpoints/w16/G.pth',
    'load_D': None,  # 'checkpoints/w16/D.pth'
    'encourage_variance': True,
    'var_coeff': 1
}

DATASET_CONFIG = {
    'window': 16,
    'len': 2048,
    'transpose': True
}
