RNN_CONFIG = {
    'num_layers_G': 2,
    'num_layers_D': 1,
    'hidden_size': 256,
    'out_channels': 16,
    'kernel_size': 8,
    'padding': 0,
    'stride': 1,
    'random_input': 100,
    'dropout_G': 0.3,
    'dropout_D': 0,
    'music_size': 89,  # 88 notes + silence
    'clamp': .01
}

TRAIN_CONFIG = {
    'batch_size': 128,
    'workers': 6,
    'K': 5,
    'epochs': 100,
    'learning_rate': 3e-2,
    'weight_decay': 0,
    'experiment_name': 'W-C-RNN',
    'balance': False,
    'feature_matching': False,
    'save_every_n_epochs': 5,
    'load_G': None,  # 'checkpoints/w16/G.pth',
    'load_D': None,  # 'checkpoints/w16/D.pth'
    'encourage_variance': False,
    'var_coeff': 10,
    'wasserstein': True,
    'noise_D_real_input': True
}

DATASET_CONFIG = {
    'window': 32,
    'len': 2048,
    'transpose': False
}
