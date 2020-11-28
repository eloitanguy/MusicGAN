import torch
from torch.utils.data import Dataset
import numpy as np
from config import DATASET_CONFIG


SONG_NAMES = [name.rstrip() for name in open("data/songnames.txt", "r").readlines()]


class TouhouDataset(Dataset):
    def __init__(self):
        self.midi_arrays = []
        for song_name in SONG_NAMES:
            song_path = 'data/dual/' + song_name + '.npy'
            a = np.load(song_path)
            self.midi_arrays.append(a)

    def __len__(self):
        return DATASET_CONFIG['len']

    def __getitem__(self, idx):
        song_idx = idx % len(self.midi_arrays)
        song = self.midi_arrays[song_idx]
        t_max = song.shape[0] - DATASET_CONFIG['window']
        t_0 = np.random.randint(0, t_max)
        t_end = t_0 + DATASET_CONFIG['window']
        return torch.tensor(song[t_0:t_end, :]).float()
