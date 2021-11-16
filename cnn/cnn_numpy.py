import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


def initialize_conv_weights(n, k, d):
  """
  n = number of filters
  k = filter width (same as height)
  d = filter depth

  Returns:
  np.ndarray: (# of filters, filter height, filter width, filter depth)
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


def linear(X, W, b):
  return X @ W + b


def conv(X, W, b, stride=1):
  """
  X = input to be convolved (assumed to be square)
  W = filters (assumed to be square)
  b = biases (assumed to be "tied": one bias per feature map)
  """
  num_filters = W.shape[0]
  out_depth = num_filters
  filter_width = W.shape[1]
  out_width = (X.shape[0] - filter_width) // stride + 1
  out_volume = np.empty((out_width, out_width, out_depth))
  for k, w in enumerate(W):
    w = w.squeeze()
    for i in range(out_width):
      for j in range(out_width):
        out_volume[i, j, k] = np.sum(
          X[i*stride:i*stride+filter_width, j*stride:j*stride+filter_width] * w) + b[k]
  return out_volume


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


def conv_derivative(X, dV, W):
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

    self.W0 = initialize_conv_weights(4, 8, 1)
    self.b0 = np.zeros(4)

    self.W1 = initialize_conv_weights(8, 4, 4)
    self.b1 = np.zeros(8)

    self.W2 = initialize_fc_weights(8 * 8 * 8, 16)
    self.b2 = np.zeros(16)

    self.W3 = initialize_fc_weights(16, 10)
    self.b3 = np.zeros(10)

  def predict(self, x):
    output = []
    for img in x:
      V0 = conv(img, self.W0, self.b0, stride=2)
      z0 = relu(V0)
      V1 = conv(z0, self.W1, self.b1, stride=1)
      z1 = relu(V1.flatten())
      h0 = linear(z1, self.W2, self.b2)
      z2 = relu(h0)
      h1 = linear(z2, self.W3, self.b3)
      output.append(softmax(h1).argmax())
    return np.array(output)

  def train(self, x, y, y_one_hot):
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
      V0 = conv(img, self.W0, self.b0, stride=2)
      z0 = relu(V0)

      V1 = conv(z0, self.W1, self.b1, stride=1)
      z1 = relu(V1.flatten())

      h0 = linear(z1, self.W2, self.b2)
      z2 = relu(h0)

      h1 = linear(z2, self.W3, self.b3)
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

      dW1, dz0 = conv_derivative(z0, dV1, self.W1)
      db1 = dV1.mean(axis=(0, 1))

      dV0 = dz0 * relu_derivative(z0)
      dW0, dimg = conv_derivative(img, dV0, self.W0)
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

  def save_parameters(self):
    with open('numpy_model.npz', 'wb') as f:
      np.savez(f,
        W0=self.W0, b0=self.b0,
        W1=self.W1, b1=self.b1,
        W2=self.W2, b2=self.b2,
        W3=self.W3, b3=self.b3)


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
  losses = []

  # train
  for x_batch, y_batch in tqdm(train_loader):
    x_batch = x_batch.numpy().squeeze()
    y_batch = y_batch.numpy().squeeze()

    one_hot_target = one_hot_encode(y_batch)
    loss = model.train(x_batch, y_batch, one_hot_target)
    model.step()
    losses.append(loss)

  fig, ax = plt.subplots()
  ax.plot(losses)
  plt.show()
  plt.close()

  model.save_parameters()

  # test
  correct = 0
  for x_test, y_test in tqdm(test_loader):
    x_test = x_test.numpy().squeeze()
    y_test = y_test.numpy().squeeze()
    test_predictions = model.predict(x_test)
    correct += (test_predictions == y_test).sum()

  print('accuracy:', correct * 100. / len(test_loader.dataset), '%')

if __name__ == "__main__":
  main()

