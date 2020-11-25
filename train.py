import torch
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from torchvision.utils import save_image
from config import TRAIN_CONFIG, RNN_CONFIG


class AverageMeter(object):
    def __init__(self):
        self.number = 0.
        self.sum = 0.
        self.avg = 0.
        self.last = 0.

    def update(self, value):
        self.number += 1.
        self.sum += value
        self.avg = self.sum / self.number
        self.last = value

    def reset(self):
        self.number, self.sum, self.avg, self.last = 0., 0., 0., 0.


def train_D_batch(batch, G, D, D_optimiser):
    bce = torch.nn.BCELoss()
    D_optimiser.zero_grad()
    batch_size = batch.shape[0]  # we expect a batch of [batch_size, sequence_length, input_size]

    # Train the discriminator on the true example
    d, _ = D(batch)
    loss = bce(d, torch.ones(batch_size, 1).cuda())  # the dataset examples are all real examples (1)

    # Train the discriminator on generated examples
    g = G(torch.randn(batch_size, RNN_CONFIG['random_input']).cuda())
    d, _ = D(g)
    loss = loss + bce(d, torch.zeros(batch_size, 1).cuda())  # we want D to say that the examples are fake

    loss.backward()
    D_optimiser.step()
    return loss.item()


def train_G_batch_feature_matching(batch, G, D, G_optimiser):
    G_optimiser.zero_grad()
    batch_size = batch.shape[0]  # we expect a batch of [batch_size, sequence_length, input_size]
    mse = torch.nn.MSELoss()

    g = G(torch.randn(batch_size, RNN_CONFIG['random_input']).cuda())
    _, r_g = D(g)  # we take D's representation of the generated example
    _, r_t = D(batch)  # we take D's representation of the true example
    loss = mse(r_g, r_t)  # We want G to fool D, ie to have a similar representation as the true example for D

    loss.backward()
    G_optimiser.step()
    return loss.item()


def train():
    train_dataset = None  # TODO
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], num_workers=TRAIN_CONFIG['workers'],
                              shuffle=True)

    G = Generator().train().cuda()
    G_optimiser = torch.optim.Adam(G.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                                   weight_decay=TRAIN_CONFIG['weight_decay'])

    D = Discriminator().train().cuda()
    D_optimiser = torch.optim.Adam(D.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                                   weight_decay=TRAIN_CONFIG['weight_decay'])

    for epoch in range(1, TRAIN_CONFIG['epochs']+1):
        D_loss, G_loss = AverageMeter(), AverageMeter()

        for idx, b in enumerate(train_loader):
            freeze_D = D_loss.last < 0.7*G_loss.last  # if D is too good we skip this batch for it
            freeze_G = G_loss.last < 0.7*D_loss.last  # if G is too good we skip this batch for it

            if not freeze_D:
                D_loss_batch = train_D_batch(b, G, D, D_optimiser)
                D_loss.update(D_loss_batch)

            if not freeze_G and idx % TRAIN_CONFIG['K'] == 0:
                G_loss_batch = train_G_batch_feature_matching(b, G, D, G_optimiser)
                G_loss.update(G_loss_batch)

        print('[{}/{}]\tD: {}\tG: {}'.format(epoch, TRAIN_CONFIG['epochs'], D_loss.avg, G_loss.avg))

    print('Saving models ...')
    torch.save({'model': D.state_dict(), 'net_config': RNN_CONFIG, 'train_config': TRAIN_CONFIG}, 'D.pth')
    torch.save({'model': G.state_dict(), 'net_config': RNN_CONFIG, 'train_config': TRAIN_CONFIG}, 'D.pth')


if __name__ == '__main__':
    train()
