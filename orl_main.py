# import jax
# import jax.numpy as jnp
import pickle
from timeit import default_timer

import numpy as np
import torch
import torchvision
import timm
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from torch import einsum
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from loguru import logger
from dataset.ORL import ORL
import nni

tensor_list = ['data/lfw-py/train_data.pt',
               'data/lfw-py/train_label.pt',
               'data/lfw-py/test_data.pt',
               'data/lfw-py/test_label.pt']


def read_dataset():
    train_data = torch.load('data/lfw-py/train_data.pt')
    train_label = torch.load('data/lfw-py/train_label.pt')
    test_data = torch.load('data/lfw-py/test_data.pt')
    test_label = torch.load('data/lfw-py/test_label.pt')
    return train_data, train_label, test_data, test_label


def chw_hwc(data):
    if data.ndim == 4:
        return np.einsum('nchw -> nhwc', data)
    elif data.ndim == 3:
        return np.einsum('chw -> hwc', data)
    else:
        raise ValueError('data must be 3 or 4 dim')


class LFWPeopleMain:
    def __init__(self):
        self.train_data, self.train_label, self.test_data, self.test_label = read_dataset()
        # self.train_mean_face = torch.mean(self.train_data, dim=0)
        # self.train_std_face = torch.std(self.train_data, dim=0)

    def get_train_data(self, size=1024):
        return self.train_data[:size].cuda(), self.train_label[:size].cuda()

    def get_test_data(self, size=256):
        return self.test_data[:size].cuda(), self.test_label[:size].cuda()

    # def get_train_centered_face(self, size=1024):
    #     return self.train_data[:size] - self.train_mean_face


def pca(X):
    """Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with most important dimensions first).
    """
    # get dimensions
    num_data, dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick used
        M = np.dot(X, X.T)  # covariance matrix
        e, EV = np.linalg.eigh(M)  # eigenvalues and eigenvectors
        tmp = np.dot(X.T, EV).T  # this is the compact trick
        V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
        S = np.sqrt(e[::-1])  # reverse since eigenvalues are in increasing order

        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]  # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V, S, mean_X


class KLT:
    def __init__(self, n_components: int = None, energy: float = None, transform: str = 'cov'):
        assert n_components is not None or energy is not None, 'n_components or energy must be specified'
        assert n_components is None or energy is None, 'n_components and energy cannot be specified at the same time'
        self.eigen_faces = None
        self.eigen_vectors = None
        self.eigen_values = None
        self.proj_mat = None
        self.mean = None
        self.n_components = n_components
        self.energy = energy

    def _set_nc_by_energy(self, eigen_values):
        if self.n_components is None:
            eigen_values_sum = eigen_values.sum()
            components_count = 0  # that they include approx. 85% of the energy
            current_energy = 0.0
            for evalue in eigen_values:
                components_count += 1
                current_energy += evalue / eigen_values_sum

                if current_energy >= self.energy:
                    self.n_components = components_count
                    break
            logger.info(f'evalues count: {components_count}, energy: {current_energy}')

    def _fit_eig(self, x: torch.Tensor) -> None:
        r"""
        Fit the PCA model with X using Eigen Decomposition
        :param x: (N, D)
        :return:
        """
        logger.info('fitting with ED')
        r = x @ x.T / x.shape[0]
        self.mean = torch.mean(x, dim=0)
        x = x - self.mean
        eigen_values, eigen_vectors = torch.linalg.eig(r)
        eigen_values = eigen_values.real
        eigen_vectors = eigen_vectors.real

        # sort eigen_values and eigenvectors
        sort_indices = torch.argsort(eigen_values, descending=True)  # getting their correct order - decreasing
        eigen_values = eigen_values[sort_indices]  # putting the eigen_values in that order
        eigen_vectors = eigen_vectors[:, sort_indices]

        # get right eigenvectors and normalize them
        eigen_vectors = x.T @ eigen_vectors  # left multiply to get the correct eigenvectors
        eigen_vectors = eigen_vectors / torch.linalg.norm(eigen_vectors, dim=0)  # normalize all eigenvectors

        # select the number of components according to the energy
        self._set_nc_by_energy(eigen_values)

        self.proj_mat = eigen_vectors[:, :self.n_components]
        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors
        self.eigen_faces = eigen_vectors.T

    def _fit_svd(self, x: torch.Tensor) -> None:
        r"""
        Fit the PCA model with X using Singular Value Decomposition
        :param x: (N, D)
        :return:
        """
        logger.info('fitting with SVD')
        n = x.shape[0]
        self.mean = torch.mean(x, dim=0)
        x = x - self.mean

        U, S, V = torch.linalg.svd(x)
        V = V[:n]  # only makes sense to return the first num_data
        S = S[:n]

        self._set_nc_by_energy(S)

        self.proj_mat = V[:self.n_components].T
        self.eigen_values = S
        self.eigen_vectors = V
        self.eigen_faces = V

    def fit(self, x: torch.Tensor) -> None:
        # if x.shape[1] > x.shape[0]:
        #     self.fit_eig(x)
        # else:
        #     self.fit_svd(x)
        self._fit_svd(x)
        # self._fit_eig(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Transform x with the fitted model
        :param x: (N, D)
        :return:
        """
        x = x - self.mean
        return x @ self.proj_mat

    def __repr__(self):
        return f'KLT(n_components={self.n_components}, energy={self.energy})'


if __name__ == '__main__':
    logger.info('start')

    parameters = nni.get_next_parameter()

    logger.info(f'parameters: {parameters}')

    hw = parameters['hw']
    energy = parameters['energy']
    model_name = parameters['model_name']

    # hw = 100
    # # n_components = 1000
    # energy = 0.999
    # n_neighbors = 1
    # num_kl_samples = 2000
    plot = False

    logger.info(f'hw: {hw}, energy: {energy}, plot: {plot}')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((hw, hw)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = ORL(root='data', split='train', transform=transform)
    test_dataset = ORL(root='data', split='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_label = train_loader.__iter__().next()
    test_data, test_label = test_loader.__iter__().next()
    train_data, train_label = train_data.cuda(), train_label.cuda()
    test_data, test_label = test_data.cuda(), test_label.cuda()
    logger.info(f'Data loaded. train data size: {train_data.shape}, test data size: {test_data.shape}')

    kl_model = KLT(energy=energy)

    logger.info(f'KL model created, model: {kl_model}, start fitting...')
    kl_model.fit(train_data.reshape(-1, hw * hw * 1))
    logger.info(f'KL model fitted, model: {kl_model}, start transforming...')

    train_data_pca = kl_model.transform(train_data.reshape(-1, hw * hw * 1))
    test_data_pca = kl_model.transform(test_data.reshape(-1, hw * hw * 1))

    logger.info(
        f'Data transformed. transformed train data size: {train_data_pca.shape}, transformed test data size: {test_data_pca.shape}')

    if plot:
        to_img = torchvision.transforms.ToPILImage()

        train_data_mean = torch.mean(train_data, dim=0)
        refactored_train_data = einsum('nk, kd -> nd',
                                       train_data_pca,
                                       kl_model.eigen_faces[:kl_model.n_components]
                                       ) + train_data_mean.reshape(-1)
        plt.figure(dpi=300, figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(train_data[0].reshape(hw, hw).cpu().numpy(), cmap='gray')
        plt.title("Original image")

        plt.subplot(2, 2, 2)
        plt.imshow(refactored_train_data[0].reshape(hw, hw).cpu().numpy(), cmap="gray")
        plt.title("Reconstructed image")

        plt.subplot(2, 2, 3)
        plt.imshow(train_data_mean.reshape(hw, hw).cpu().numpy(), cmap="gray")
        plt.title("Mean image")

        plt.subplot(2, 2, 4)
        plt.imshow(kl_model.eigen_faces[:kl_model.n_components].sum(dim=0).reshape(hw, hw).cpu().numpy(), cmap="gray")
        plt.title(f"Sum of eigvec[:{kl_model.n_components}]")
        plt.show()

        plt.figure(dpi=300, figsize=(12, 12))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(kl_model.eigen_faces[i].reshape(hw, hw).cpu().numpy(), cmap="gray")
            plt.title(f"Eigen vector[{i}]")
        plt.show()

    if model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'])
    elif model_name == 'svm':
        model = SVC(kernel='rbf', C=parameters['c'], gamma=parameters['gamma'])
    elif model_name == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=parameters['hidden_layer_sizes'],
                              learning_rate_init=parameters['learning_rate_init'])
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    # model = KNeighborsClassifier(n_neighbors=n_neighbors)
    logger.info(f'Model: {model}')

    model.fit(train_data_pca.cpu().numpy(), train_label.cpu().numpy().reshape(-1, ))
    logger.info(f"Train accuracy: {model.score(train_data_pca.cpu().numpy(), train_label.cpu().numpy().ravel()):.4f}")
    logger.info(f"Test accuracy: {model.score(test_data_pca.cpu().numpy(), test_label.cpu().numpy().ravel()):.4f}")
    nni.report_final_result(model.score(test_data_pca.cpu().numpy(), test_label.cpu().numpy().ravel()))

    d = 3
