from os.path import join as pjoin

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils.dataloader import load_dataset
from model.vae import VAE


def main():

    if not os.path.exists('results_onebyone'):
        os.makedirs('results_onebyone')

    target_type = 0

    data_generator = load_dataset(load_dir='data/test_data.csv', target_type=target_type,
                                  batch_size=16, train_data=False)
    encoder = VAE()
    encoder.load_model('model/model_8000.pt')

    label_all, z_all = [], []
    for n_iter, (x, label) in enumerate(data_generator):
        with torch.no_grad():
            z_dist = encoder.encode(x)
            z_all.append(z_dist.loc.numpy())
            label_all.append(label.numpy())
    label_all = np.concatenate(label_all)
    z_all = np.concatenate(z_all)

    z_minmax1 = z_all[:, 0].min() - 0.02, z_all[:, 0].max() + 0.02
    z_minmax2 = z_all[:, 1].min() - 0.02, z_all[:, 1].max() + 0.02

    label_all, z_all = [], []
    for n_iter, (x, label) in enumerate(data_generator):
        with torch.no_grad():
            z_dist = encoder.encode(x)
            z_all.append(z_dist.loc.numpy())
            label_all.append(label.numpy())

            z_current = z_dist.loc.numpy()

        plt.scatter(np.concatenate(z_all)[:, 0], np.concatenate(z_all)[:, 1], s=1.0)
        plt.scatter(z_current[:, 0], z_current[:, 1], s=1.0)
        plt.xlim(z_minmax1)
        plt.ylim(z_minmax2)
        plt.savefig(pjoin('results_onebyone', f'latent_space_{n_iter}_test.png'), dpi=300)
        plt.close('all')


if __name__ == '__main__':
    main()