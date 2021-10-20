from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import intel_pytorch_extension as ipex # .to(ipex.DEVICE)


# reference: https://nextjournal.com/gkoehler/pytorch-mnist


class SimpleNet(nn.Module):

  def __init__(self):
    super().__init__()
    # 28x28x1 -> 11x11x4
    self.conv1 = nn.Conv2d(1, 4, kernel_size=8, stride=2)
    # 11x11x4 -> 8x8x8
    self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=1)
    self.fl1 = nn.Linear(8 * 8 * 8, 16)
    # output 10 nodes for the 10 possible digitis [0 - 9]
    self.fl2 = nn.Linear(16, 10)

  def forward(self, x):
    h = F.relu(self.conv1(x))
    h = F.relu(self.conv2(h))
    h = F.relu(self.fl1(h.flatten(1)))
    return self.fl2(h)


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
  
  simple_net = SimpleNet()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(simple_net.parameters(), lr=learning_rate)

  # train
  for x_batch, y_batch in tqdm(train_loader):
    predictions = simple_net(x_batch)
    loss = criterion(predictions, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # test
  simple_net.eval()
  correct = 0
  with torch.no_grad():
    for x_test, y_test in tqdm(test_loader):
      test_predictions = simple_net(x_test).argmax(1)
      correct += (test_predictions == y_test).sum().item() 

  print('accuracy:', correct * 100. / len(test_loader.dataset))


if __name__ == '__main__':
  main()

