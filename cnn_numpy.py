import math
from tqdm import tqdm
import numpy as np
import torch
import torchvision


def initialize_conv_weights(k, d, n):
  """
  k = kernel size
  d = input depth
  n = output depth
  """
  rng = np.random.default_rng()
  return rng.uniform(-1/math.sqrt(k*k), 1/math.sqrt(k*k), (n,k,k,d))


def initialize_fc_weights(m, n):
  """
  m = number of inputs
  n = number of outputs
  """
  rng = np.random.default_rng()
  return rng.uniform(-math.sqrt(6/(m+n)), math.sqrt(6/(m+n)), (m,n))


def relu(x):
  return np.maximum(x, 0)


# https://deepnotes.io/softmax-crossentropy
def softmax(x):
  exps = np.exp(x - x.max())
  return exps / exps.sum()


def cross_entropy(x, y, y_one_hot):
  loss = -np.log(x[y])
  grad = x - y_one_hot
  return loss, grad


def one_hot_encode(y):
  n = len(y)
  encoded = np.zeros((n, 10))
  encoded[np.arange(n), y] = 1
  return encoded


class NumpyNet:

  def __init__(self):
    self.W0 = initialize_conv_weights(8, 1, 4)
    self.b0 = np.zeros(4)
    self.W1 = initialize_conv_weights(4, 4, 8)
    self.b1 = np.zeros(8)
    self.W2 = initialize_fc_weights(8 * 8 * 8, 16)
    self.b2 = np.zeros(16)
    self.W3 = initialize_fc_weights(16, 10)
    self.b3 = np.zeros(10)

  def forward(self, x, y, y_one_hot):
    loss = 0
    grad = np.zeros(10)
    for idx, img in enumerate(x):
      V0 = relu(self.conv1(img))
      V1 = relu(self.conv2(V0)).flatten()
      h0 = relu(self.linear1(V1))
      output = softmax(self.linear2(h0))
      sample_loss, sample_grad = cross_entropy(output, y[idx], y_one_hot[idx])
      loss += sample_loss
      grad += sample_grad
    n = len(x)
    loss /= n
    grad /= n
    return loss, grad

  def conv1(self, img):
    out_volume = np.empty((11, 11, 4))
    for activation_map_idx, w in enumerate(self.W0):
      w = w.squeeze()
      bias = self.b0[activation_map_idx]
      kernel_size = w.shape[0]
      for i in range(11):
        for j in range(11):
          out_volume[i, j, activation_map_idx] = np.sum(
            img[i*2:i*2+kernel_size, j*2:j*2+kernel_size] * w) + bias
    return out_volume

  def conv2(self, V):
    out_volume = np.empty((8, 8, 8))
    for activation_map_idx, w in enumerate(self.W1):
      w = w.squeeze()
      bias = self.b1[activation_map_idx]
      kernel_size = w.shape[0]
      for i in range(8):
        for j in range(8):
          out_volume[i, j, activation_map_idx] = np.sum(
            V[i:i+kernel_size, j:j+kernel_size] * w) + bias
    return out_volume

  def linear1(self, V):
    return V @ self.W2 + self.b2

  def linear2(self, V):
    return V @ self.W3 + self.b3

  def backward(self, grad):
    pass


def main():
  batch_size_train = 64
  batch_size_test = 1000
  learning_rate = 0.01

  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('~/datasets', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('~/datasets', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

  model = NumpyNet()

  # train
  for x_batch, y_batch in tqdm(train_loader):
    x_batch = x_batch.numpy().squeeze()
    y_batch = y_batch.numpy().squeeze()

    one_hot_target = one_hot_encode(y_batch)
    loss, grad = model.forward(x_batch, y_batch, one_hot_target)

    return


if __name__ == "__main__":
  main()

