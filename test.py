from models import Generator, Discriminator
import torch
from dataset import TouhouDataset
from config import DATASET_CONFIG, TRAIN_CONFIG
from data.midi import arry2mid, plot_midi

exp_name = TRAIN_CONFIG['experiment_name']
length = 256

checkpoint_G = torch.load('checkpoints/{}/G.pth'.format(exp_name))
G = Generator(checkpoint_G['net_config']).eval()
G.load_state_dict(checkpoint_G['model'])

checkpoint_D = torch.load('checkpoints/{}/D.pth'.format(exp_name))
D = Discriminator(checkpoint_D['net_config']).eval()
D.load_state_dict(checkpoint_D['model'])

z = torch.randn(1, DATASET_CONFIG['window'], 100)
z[:, :, ::4] = 4*z[:, :, ::4]
g = G(z)
print('max G', g.max().item())
print('D output on g', torch.sigmoid(D(g)[0]).item())

data = TouhouDataset()
mse = torch.nn.MSELoss()

for music in data:
    d_t, r_t = D(music.unsqueeze(0))
    _, r_g = D(g)
    print('D on true data', torch.sigmoid(d_t).item())
    print('D representation difference', mse(r_t, r_g).item())
    break

z = torch.randn(1, length, 100)
z[:, :, ::4] = 4*z[:, :, ::4]
g = G(z)

notes = g.squeeze().clone().detach().numpy()
notes[notes > 0.5] = 1
notes[notes != 1] = 0
mid = arry2mid(notes.astype(int))
mid.save('g.mid')
plot_midi(notes)
