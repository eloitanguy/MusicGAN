RNN_CONFIG = {
    'num_layers': 2,  # paper value
    'hidden_size': 350,  # paper value
    'random_input': 100,
    'dropout': 0.2,
    'music_size': 89
}

TRAIN_CONFIG = {
    'batch_size': 256,
    'workers': 6,
    'K': 1,
    'epochs': 10,
    'learning_rate': 1e-1,  # paper
    'weight_decay': 1e-4
}
