from models import Generator
import torch
from config import TRAIN_CONFIG
from data.midi import arry2mid, plot_midi
import numpy as np
import argparse


if __name__ == '__main__':
    exp_name = TRAIN_CONFIG['experiment_name']
    parser = argparse.ArgumentParser(description='Music generator from trained models')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/{}/G.pth'.format(exp_name),
                        help='path to the G checkpoint file, default: use the final checkpoint of the last training.')
    args = parser.parse_args()
    length = 256

    checkpoint_G = torch.load(args.checkpoint)
    G = Generator(checkpoint_G['net_config']).eval()
    G.load_state_dict(checkpoint_G['model'])

    z = torch.randn(1, length, 100)
    z[:, :, ::4] = 4*z[:, :, ::4]
    g = G(z)

    notes = g.squeeze().clone().detach().numpy()
    note_t = np.quantile(notes, 87/89)  # tries to have 2 notes on average
    output = notes.copy()
    output[notes >= note_t] = 1
    output[notes < note_t] = 0
    mid = arry2mid(output.astype(int))
    mid.save('g.mid')
    plot_midi(output)
