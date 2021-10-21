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


def relu_derivative(x):
  out = np.zeros(*x.shape) if x.ndim == 1 else np.zeros(x.shape)
  out[x > 0] = 1
  return out


def conv2_backward(X, dV, W):
  """
  X = input volume that was convolved during the forward pass
  dV = gradient wrt output volume
  W = filter set that we are calculating the derivative of
  """
  if X.ndim < 3:
    X = X[..., np.newaxis]
  I, J, K = dV.shape
  dW = np.zeros(W.shape)
  dX = np.zeros(X.shape)
  fh, fw, fd = W.shape[1:]
  for k in range(K):
    for i in range(I):
      for j in range(J):
        dW[k] += np.outer(X[i:i+fh, j:j+fw, :], dV[i, j, k]).reshape(fh, fw, fd)
        dX[i:i+fh, j:j+fw, :] += np.outer(W[k], dV[i, j, k]).reshape(fh, fw, fd)
  return dW, dX


class NumpyNet:

  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

    self.W0 = initialize_conv_weights(8, 1, 4)
    self.b0 = np.zeros(4)

    self.W1 = initialize_conv_weights(4, 4, 8)
    self.b1 = np.zeros(8)

    self.W2 = initialize_fc_weights(8 * 8 * 8, 16)
    self.b2 = np.zeros(16)

    self.W3 = initialize_fc_weights(16, 10)
    self.b3 = np.zeros(10)

  def forward(self, x, y, y_one_hot):
    loss_average = 0
    self.dW3_average = np.zeros((16, 10))
    self.db3_average= np.zeros(10)
    self.dW2_average = np.zeros((8 * 8 * 8, 16))
    self.db2_average = np.zeros(16)
    self.dW1_average = np.zeros((8, 4, 4, 4))
    self.db1_average = np.zeros(8)
    self.dW0_average = np.zeros((4, 8, 8, 1))
    self.db0_average = np.zeros(4)

    for idx, img in enumerate(x):
      # forward
      V0 = self.conv1(img)
      z0 = relu(V0)

      V1 = self.conv2(z0)
      z1 = relu(V1.flatten())

      h0 = self.linear1(z1)
      z2 = relu(h0)

      h1 = self.linear2(z2)
      output = softmax(h1)

      loss, dh1 = cross_entropy(output, y[idx], y_one_hot[idx])

      # backward
      loss_average += loss

      dW3 = np.outer(z2, dh1)
      db3 = dh1

      dz2 = dh1 @ self.W3.T
      dh0 = dz2 * relu_derivative(h0)

      dW2 = np.outer(z1, dh0)
      db2 = dh0

      dz1 = (dh0 @ self.W2.T).reshape(8, 8, 8)
      dV1 = dz1 * relu_derivative(V1)

      dW1, dz0 = conv2_backward(z0, dV1, self.W1)
      db1 = dV1.mean(axis=(0, 1))

      dV0 = dz0 * relu_derivative(z0)
      dW0, dimg = conv2_backward(img, dV0, self.W0)
      db0 = dV0.mean(axis=(0, 1))

      self.dW3_average += dW3
      self.db3_average += db3
      self.dW2_average += dW2
      self.db2_average += db2
      self.dW1_average += dW1
      self.db1_average += db1
      self.dW0_average += dW0
      self.db0_average += db0

    n = len(x)
    loss_average /= n
    self.dW3_average /= n
    self.db3_average /= n
    self.dW2_average /= n
    self.db2_average /= n
    self.dW1_average /= n
    self.db1_average /= n
    self.dW0_average /= n
    self.db0_average /= n

    return loss_average

  def step(self):
    self.W0 -= self.learning_rate * self.dW0_average
    self.b0 -= self.learning_rate * self.db0_average

    self.W1 -= self.learning_rate * self.dW1_average
    self.b1 -= self.learning_rate * self.db1_average

    self.W2 -= self.learning_rate * self.dW2_average
    self.b2 -= self.learning_rate * self.db2_average

    self.W3 -= self.learning_rate * self.dW3_average
    self.b3 -= self.learning_rate * self.db3_average

  def conv1(self, img):
    out_volume = np.empty((11, 11, 4))
    for k, w in enumerate(self.W0):
      w = w.squeeze()
      bias = self.b0[k]
      kernel_size = w.shape[0]
      for i in range(11):
        for j in range(11):
          out_volume[i, j, k] = np.sum(
            img[i*2:i*2+kernel_size, j*2:j*2+kernel_size] * w) + bias
    return out_volume

  def conv2(self, V):
    out_volume = np.empty((8, 8, 8))
    for k, w in enumerate(self.W1):
      w = w.squeeze()
      bias = self.b1[k]
      kernel_size = w.shape[0]
      for i in range(8):
        for j in range(8):
          out_volume[i, j, k] = np.sum(
            V[i:i+kernel_size, j:j+kernel_size] * w) + bias
    return out_volume

  def linear1(self, V):
    return V @ self.W2 + self.b2

  def linear2(self, V):
    return V @ self.W3 + self.b3


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

  model = NumpyNet(learning_rate)

  # train
  for x_batch, y_batch in tqdm(train_loader):
    x_batch = x_batch.numpy().squeeze()
    y_batch = y_batch.numpy().squeeze()

    one_hot_target = one_hot_encode(y_batch)
    loss = model.forward(x_batch, y_batch, one_hot_target)


if __name__ == "__main__":
  main()

