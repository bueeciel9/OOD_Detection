from os.path import join as pjoin

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.dataloader import load_dataset
from model.vae import VAE


def dist_matrix(data):
    sq_dists = np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :])**2, axis=-1)  # squared pairwise distances
    distances = np.sqrt(sq_dists)  # pairwise distances
    return distances


def main():

    if not os.path.exists('results_ood'):
        os.makedirs('results_ood')

    data_generator = load_dataset(load_dir='data/train_data_.csv', batch_size=5000, train_data=False)
    data_generator_test = load_dataset(load_dir='data/test_data_.csv', batch_size=10000, train_data=False)
    encoder = VAE()
    encoder.load_model('model/model_8000.pt')

    data_answer = pd.read_csv('data/answer_sample.csv').to_numpy()
    data_answer = data_answer[:, 1]
    data_answer[:] = 0

    for i in range(8):
        if i != 1:
            for x, label in data_generator:
                with torch.no_grad():
                    z_dist = encoder.encode(x)
                    label_train = label.numpy() == i
                    z_train = z_dist.loc.numpy()[label_train]
                break

            for x, label in data_generator_test:
                with torch.no_grad():
                    z_dist = encoder.encode(x)
                    label_test = label.numpy() == i
                    z_test = z_dist.loc.numpy()[label_test]
                break

            xn = np.linspace(z_train[:, 0].min(), z_train[:, 0].max(), 100)
            yn = np.poly1d(np.polyfit(z_train[:, 0], z_train[:, 1], deg=3))
            plt.scatter(z_train[:, 0], z_train[:, 1], s=1.0)
            plt.plot(xn, yn(xn), color='red')
            plt.scatter(z_test[:, 0], z_test[:, 1], s=1.0, color='green')
            xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
            plt.savefig(f'results_ood/{i}_regression.png')
            plt.close('all')

            diff = np.abs((z_test[:, 1] - yn(z_test[:, 0])))
            counts, bins = np.histogram(diff, bins=20)
            plt.hist(bins[:-1], bins, weights=counts)
            plt.savefig(f'results_ood/{i}_hist.png')
            plt.close('all')
            ood = diff > 0.02
            label_test[label_test] = ood
            data_answer[label_test] = 1

            plt.scatter(z_train[:, 0], z_train[:, 1], s=1.0)
            plt.plot(xn, yn(xn), color='red')
            plt.scatter(z_test[:, 0], z_test[:, 1], s=1.0, color='green')
            plt.scatter(z_test[:, 0][ood], z_test[:, 1][ood], s=1.0, color='blue')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.savefig(f'results_ood/{i}_regression_ood.png')
            plt.close('all')

            plt.scatter(z_test[:, 0][ood], z_test[:, 1][ood], s=1.0, color='blue')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.savefig(f'results_ood/{i}_regression_ood_only.png')
            plt.close('all')

    data = pd.read_csv('data/answer_sample.csv')
    data['label'] = data_answer
    data.to_csv('results_ood/answer.csv', index=False)


if __name__ == '__main__':
    main()