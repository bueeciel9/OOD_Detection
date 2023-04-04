import os
from os.path import join as pjoin

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils.dataloader import load_dataset
from model.vae import VAE


def pairwise_mse_loss(tensors):
    n = len(tensors)  # number of tensors
    mse_loss = torch.zeros(n, n)  # initialize loss matrix

    # calculate pairwise MSE loss
    for i in range(n):
        for j in range(i, n):
            mse_loss[i, j] = mse_loss[j, i] = torch.nn.functional.mse_loss(tensors[i], tensors[j])

    return mse_loss.mean()


def main():

    if not os.path.exists('results'):
        os.makedirs('results')

    plot_with_sample = False

    data_generator = load_dataset(load_dir='data/train_data_.csv', batch_size=5000, target_type=None)
    data_generator_test = load_dataset(load_dir='data/test_data_.csv', batch_size=10000, target_type=None)
    encoder = VAE()
    optimizer = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-6)

    for epoch in range(20000):

        if epoch > 0:
            loss = {
                'VAE Loss': 0.0,
                'Recon Loss': 0.0,
                'KL Loss': 0.0
            }

            for n_iter, (x, label) in enumerate(data_generator):

                ###### get z distribution & reparametrize trick: z ~ p(z|x)
                z_dist = encoder.encode(x)
                # z = z_dist.rsample()
                z = z_dist.loc

                ###### decode: p(x|z)
                x_pred = encoder.decode(z)

                ###### compute loss & update model
                encoder.zero_grad()
                optimizer.zero_grad()
                loss_latent, loss_recon, loss_kl = encoder.loss(z_dist, x_pred, x)

                # z_reg_all = []
                # for i in range(8):
                #     if i != 1:
                #         z_reg_all.append(torch.cat([
                #             z[label == i].mean(0),
                #             z[label == i].min(0).values,
                #             z[label == i].max(0).values,
                #         ]))
                # regularization = pairwise_mse_loss(torch.stack(z_reg_all))
                # loss_latent += regularization

                loss_latent.backward()
                optimizer.step()

                loss['VAE Loss'] += loss_latent.item()
                loss['Recon Loss'] += loss_recon.item()
                loss['KL Loss'] += loss_kl.item()

            print(f'Epoch: {epoch} | ' + ' | '.join([f'{key}: {item / (n_iter + 1):.4f}' for key, item in loss.items()]))

        if epoch % 1000 == 0:
            label_all, z_all = [], []
            for n_iter, (x, label) in enumerate(data_generator):
                with torch.no_grad():
                    z_dist = encoder.encode(x)
                    if plot_with_sample:
                        z_all.append(z_dist.sample().numpy())
                    else:
                        z_all.append(z_dist.loc.numpy())
                    label_all.append(label.numpy())
            label_all = np.concatenate(label_all)
            z_all = np.concatenate(z_all)

            label_all_train = label_all.copy()
            z_all_train = z_all.copy()

            plt.scatter(z_all[:, 0], z_all[:, 1], s=1.0, c=label_all, cmap='coolwarm')
            plt.colorbar()
            xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
            plt.savefig(pjoin('results', f'latent_space_{epoch}.png'))
            plt.close('all')

            for i in range(8):
                label_target = label_all == i

                plt.scatter(z_all[label_target][:, 0],
                            z_all[label_target][:, 1], s=1.0,
                            c=label_all[label_target], cmap='coolwarm',
                            vmin=0, vmax=7)
                plt.colorbar()
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.savefig(pjoin('results', f'latent_space_{epoch}_{i}.png'))
                plt.close('all')

            label_all, z_all = [], []
            for n_iter, (x, label) in enumerate(data_generator_test):
                with torch.no_grad():
                    z_dist = encoder.encode(x)
                    if plot_with_sample:
                        z_all.append(z_dist.sample().numpy())
                    else:
                        z_all.append(z_dist.loc.numpy())
                    label_all.append(label.numpy())
            label_all = np.concatenate(label_all)
            z_all = np.concatenate(z_all)

            plt.scatter(z_all[:, 0], z_all[:, 1], s=1.0, c=label_all, cmap='coolwarm')
            plt.colorbar()
            xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
            plt.savefig(pjoin('results', f'latent_space_{epoch}_test.png'))
            plt.close('all')

            for i in range(8):
                label_target = label_all == i

                plt.scatter(z_all[label_target][:, 0],
                            z_all[label_target][:, 1], s=1.0,
                            # c=label_all[label_target],
                            # cmap='coolwarm', vmin=0, vmax=7
                            )

                plt.scatter(
                    z_all_train[label_all_train == i][:, 0],
                    z_all_train[label_all_train == i][:, 1], s=1.0,
                    # c=label_all_train[label_all_train == i],
                    # cmap='coolwarm', vmin=0, vmax=7
                )

                # plt.xlim(xlim)
                # plt.ylim(ylim)
                plt.savefig(pjoin('results', f'latent_space_{epoch}_test_{i}.png'))
                plt.close('all')

            encoder.save_model(f'results/model_{epoch}.pt')
    print('hi')


if __name__ == "__main__":
    main()
