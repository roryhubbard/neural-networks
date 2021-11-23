import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import create_masks, additive_subsequent_mask, make_model


def make_copy_task(nbatches, batch_size, vocab_size):
  samples = []
  rng = np.random.default_rng()
  for _ in range(nbatches * batch_size):
    x = rng.integers(low=0, high=vocab_size, size=4)
    y = x.copy()
    samples.append((torch.tensor(x), torch.tensor(y)))

  train_loader = DataLoader(samples, batch_size=batch_size)
  criterion = nn.NLLLoss()
  model = make_model(10, 10, 2)

  return train_loader, criterion, model


def greedy_decode(model, src, start_symbol):
  memory = model.encode(src)
  running_output = torch.tensor([start_symbol]).unsqueeze(0)
  for _ in range(src.size(1)):
    tgt_mask = additive_subsequent_mask(running_output.size(1))
    out = model.decode(memory, running_output, 0, tgt_mask)
    prob = model.generator(out[:, -1])
    last_output = torch.argmax(prob, dim=-1).unsqueeze(0)
    running_output = torch.cat([running_output, last_output], dim=1)
  return running_output


def main():
  batch_size = 32
  nbatches = 20
  vocab_size = 4
  train_loader, criterion, model = make_copy_task(nbatches, batch_size, vocab_size)
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

  epochs = 0
  losses = []

  try:
    for _ in tqdm(range(epochs)):
      for src_batch, tgt_batch in train_loader:
        src_mask, tgt_mask = create_masks(src_batch, tgt_batch, -1, -1)

        out = model(src_batch, tgt_batch, src_mask, tgt_mask)
        loss = criterion(out.transpose(1, 2), tgt_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

  except KeyboardInterrupt:
    pass

  model.eval()
  src = torch.arange(vocab_size).unsqueeze(0)
  start_symbol = src[0][0].item()
  out = greedy_decode(model, src, start_symbol)

  print(out)


if __name__ == "__main__":
  main()

