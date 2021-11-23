import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import create_masks, additive_subsequent_mask, make_model


def make_copy_data(nbatches, batch_size, vocab_size):
  samples = []
  rng = np.random.default_rng()
  for _ in range(nbatches * batch_size):
    x = rng.integers(low=1, high=vocab_size, size=10)
    x[0] = 1
    y = x.copy()
    samples.append((torch.tensor(x), torch.tensor(y)))

  return DataLoader(samples, batch_size=batch_size)


def greedy_decode(model, src, start_symbol):
  memory = model.encode(src)
  running_output = torch.tensor([start_symbol]).unsqueeze(0)
  for _ in range(src.size(1)-1):
    tgt_mask = additive_subsequent_mask(running_output.size(1))
    out = model.decode(running_output, memory, tgt_mask)
    prob = model.generator(out[:, -1])
    last_output = torch.argmax(prob, dim=-1).unsqueeze(0)
    running_output = torch.cat([running_output, last_output], dim=1)
  return running_output


def main(custom_transformer):
  batch_size = 32
  nbatches = 32
  vocab_size = 11
  train_loader = make_copy_data(nbatches, batch_size, vocab_size)
  criterion = nn.NLLLoss()
  model = make_model(vocab_size, vocab_size, N=2, custom_transformer=custom_transformer)
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

  epochs = 10
  losses = []

  try:
    for _ in tqdm(range(epochs)):
      for src_batch, tgt_batch in train_loader:
        tgt_in = tgt_batch[:, :-1]
        tgt_out = tgt_batch[:, 1:]
        src_mask, tgt_mask, memory_mask, src_key_padding_mask, \
          tgt_key_padding_mask, memory_key_padding_mask = \
          create_masks(src_batch, tgt_in, -1, -1)

        out = model(src_batch, tgt_in, src_mask, tgt_mask, memory_mask,
                    src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        loss = criterion(out.reshape(-1, out.shape[-1]), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

  except KeyboardInterrupt:
    pass

  fig, ax = plt.subplots()
  ax.plot(losses)
  plt.show()
  plt.close()

  model.eval()
  src = torch.arange(1, vocab_size).unsqueeze(0)
  start_symbol = src[0][0].item()
  out = greedy_decode(model, src, start_symbol)

  print(src)
  print(out)


if __name__ == "__main__":
  custom_transformer = True
  main(custom_transformer)

