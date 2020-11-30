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
        self.augment = DATASET_CONFIG['transpose']

    def __len__(self):
        return DATASET_CONFIG['len']

    def __getitem__(self, idx):
        song_idx = idx % len(self.midi_arrays)
        song = self.midi_arrays[song_idx]
        t_max = song.shape[0] - DATASET_CONFIG['window']
        t_0 = np.random.randint(0, t_max)
        t_end = t_0 + DATASET_CONFIG['window']
        song_snippet = song[t_0:t_end, :]
        if self.augment:
            song_snippet = random_transposition_augmentation(song_snippet)
        return torch.tensor(song_snippet).float()


def random_transposition_augmentation(music_track_array):
    """
    Returns a transposed version of the given track
    """
    add = music_track_array.shape[1] - 88
    nonzero_notes = np.nonzero(music_track_array[:, add:])[1]
    note_min, note_max = np.min(nonzero_notes), np.max(nonzero_notes)
    t = np.random.randint(0, 88 - note_max + note_min)
    res = np.full(music_track_array.shape, 0)
    res[:, (add + t):(add + t + 1 + note_max - note_min)] = \
        music_track_array[:, (add + note_min):(add + 1 + note_max)]
    return res

