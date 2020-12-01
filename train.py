import torch
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from dataset import TouhouDataset
from config import TRAIN_CONFIG, RNN_CONFIG, DATASET_CONFIG
import os


class LossMeter(object):
    def __init__(self):
        self.number = 0.
        self.sum = 0.
        self.avg = 0.
        self.last_ratio = 0.
        self.first = 1.

    def update(self, value, first=False):
        self.number += 1.
        self.sum += value
        self.avg = self.sum / self.number
        if first:
            self.first = value
        self.last_ratio = value / (self.first + 1e-8)

    def reset(self):
        self.number, self.sum, self.avg = 0., 0., 0.


def train_D_batch(batch, G, D, D_optimiser):
    bce = torch.nn.BCEWithLogitsLoss()
    D_optimiser.zero_grad()
    batch_size = batch.shape[0]  # we expect a batch of [batch_size, sequence_length, input_size]

    # Train the discriminator on the true example
    d, _ = D(batch)
    loss = bce(d, torch.ones(batch_size).cuda())  # the dataset examples are all real examples (1)

    # Train the discriminator on generated examples
    g = G(torch.randn(batch_size, DATASET_CONFIG['window'], RNN_CONFIG['random_input']).cuda())
    d, _ = D(g)
    loss = loss + bce(d, torch.zeros(batch_size).cuda())  # we want D to say that the examples are fake

    loss.backward()
    D_optimiser.step()
    return loss.item() / 2  # over 2 because D is trained on twice as many examples without proper averaging


def train_G_batch_feature_matching(batch, G, D, G_optimiser):
    G_optimiser.zero_grad()
    batch_size = batch.shape[0]  # we expect a batch of [batch_size, sequence_length, input_size]
    mse = torch.nn.MSELoss()

    g = G(torch.randn(batch_size, DATASET_CONFIG['window'], RNN_CONFIG['random_input']).cuda())
    _, r_g = D(g)  # we take D's representation of the generated example
    _, r_t = D(batch)  # we take D's representation of the true example
    loss = mse(r_g, r_t)  # We want G to fool D, ie to have a similar representation as the true example for D

    loss.backward()
    G_optimiser.step()
    return loss.item()


def train_G_batch(batch, G, D, G_optimiser):
    G_optimiser.zero_grad()
    batch_size = batch.shape[0]  # we expect a batch of [batch_size, sequence_length, input_size]
    bce = torch.nn.BCEWithLogitsLoss()

    g = G(torch.randn(batch_size, DATASET_CONFIG['window'], RNN_CONFIG['random_input']).cuda())
    d, _ = D(g)
    loss = bce(d, torch.ones(batch_size).cuda())  # We want G to fool D

    if TRAIN_CONFIG['encourage_variance']:
        mse = torch.nn.MSELoss()
        # encouraging having a high empirical variance in terms of different notes
        loss = loss - TRAIN_CONFIG['var_coeff'] * mse(g-torch.mean(g, dim=(0, 1)))

    loss.backward()
    G_optimiser.step()
    return loss.item()


def train():
    train_dataset = TouhouDataset()
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], num_workers=TRAIN_CONFIG['workers'],
                              shuffle=True)

    if TRAIN_CONFIG['load_G']:
        G_checkpoint = torch.load(TRAIN_CONFIG['load_G'])
        G = Generator(config=G_checkpoint['net_config'])
        G.load_state_dict(G_checkpoint['model'])
        G = G.train().cuda()
    else:
        G = Generator().train().cuda()

    if TRAIN_CONFIG['load_D']:
        D_checkpoint = torch.load(TRAIN_CONFIG['load_D'])
        D = Discriminator(config=D_checkpoint['net_config'])
        D.load_state_dict(D_checkpoint['model'])
        D = D.train().cuda()
    else:
        D = Discriminator().train().cuda()

    G_optimiser = torch.optim.Adam(G.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                                   weight_decay=TRAIN_CONFIG['weight_decay'])
    G_trainer = train_D_batch if TRAIN_CONFIG['feature_matching'] else train_G_batch_feature_matching
    D_optimiser = torch.optim.Adam(D.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                                   weight_decay=TRAIN_CONFIG['weight_decay'])

    checkpoint_folder = 'checkpoints/{}/'.format(TRAIN_CONFIG['experiment_name'])
    D_loss, G_loss = LossMeter(), LossMeter()

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
        D_loss.reset()
        G_loss.reset()

        for idx, batch in enumerate(train_loader):

            # if D is too good we skip this batch for it
            freeze_D = D_loss.last_ratio < 0.7 * G_loss.last_ratio and TRAIN_CONFIG['balance'] and idx != 0
            # if G is too good we skip this batch for it
            freeze_G = G_loss.last_ratio < 0.7 * D_loss.last_ratio and TRAIN_CONFIG['balance'] and idx != 0
            x = batch.cuda()

            if not freeze_D:
                D_loss_batch = train_D_batch(x, G, D, D_optimiser)
                D_loss.update(D_loss_batch, first=idx == 0)

            if not freeze_G and idx % TRAIN_CONFIG['K'] == 0:
                G_loss_batch = G_trainer(x, G, D, G_optimiser)
                G_loss.update(G_loss_batch, first=idx == 0)

        print('[{}/{}]\tD: {:.5f}\tG: {:.5f}'.format(epoch, TRAIN_CONFIG['epochs'],
                                                     D_loss.avg if D_loss.avg != 0 else D_loss.last_ratio,
                                                     G_loss.avg if G_loss.avg != 0 else G_loss.last_ratio))

        if epoch % TRAIN_CONFIG['save_every_n_epochs'] == 0:
            torch.save({'model': D.state_dict(), 'net_config': RNN_CONFIG, 'train_config': TRAIN_CONFIG},
                       checkpoint_folder + f'D_ep{epoch}.pth')
            torch.save({'model': G.state_dict(), 'net_config': RNN_CONFIG, 'train_config': TRAIN_CONFIG},
                       checkpoint_folder + f"G_ep{epoch}.pth")

    print('Saving models ...')
    torch.save({'model': D.state_dict(), 'net_config': RNN_CONFIG, 'train_config': TRAIN_CONFIG},
               checkpoint_folder + 'D.pth')
    torch.save({'model': G.state_dict(), 'net_config': RNN_CONFIG, 'train_config': TRAIN_CONFIG},
               checkpoint_folder + 'G.pth')


if __name__ == '__main__':
    train()
