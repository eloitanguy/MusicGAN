RNN_CONFIG = {
    'num_layers': 2,  # paper value
    'hidden_size': 350,  # paper value
    'random_input': 100,
    'dropout': 0,
    'music_size': 89*2  # left hand + right hand
}

TRAIN_CONFIG = {
    'batch_size': 128,
    'workers': 6,
    'K': 1,
    'epochs': 50,
    'learning_rate': 1e-1,
    'weight_decay': 1e-4,
    'experiment_name': 'w8',
    'balance': True,
    'feature_matching': True,
    'save_every_n_epochs': 5,
    'load_G': None,  # 'checkpoints/w8/G.pth',
    'load_D': None  # 'checkpoints/w8/D.pth'
}

DATASET_CONFIG = {
    'window': 8,
    'len': 2048,
    'transpose': True
}
