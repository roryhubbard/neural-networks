import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IWSLT2017

from transformer import Transformer
from utils import create_masks, make_model


def text_to_tensor(text, vocab, tokenizer):
  return torch.cat((torch.tensor([vocab['<bos>']]),
                   torch.tensor(vocab(tokenizer(text))),
                   torch.tensor([vocab['<eos>']])))


def yield_tokens(data_iter, tokenizer, sample_idx):
  for data_sample in data_iter:
    yield tokenizer(data_sample[sample_idx])


def get_vocabs_and_tokenizer():
  src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  tgt_tokenizer = get_tokenizer('spacy', language='nl_core_news_sm')
  specials = ['<unk>', '<pad>', '<bos>', '<eos>']

  train_iter = IWSLT2017(root='/home/chub/datasets',
                         split='train', language_pair=('en', 'nl'))
  src_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, src_tokenizer, 0), specials=specials)
  src_vocab.set_default_index(src_vocab['<unk>'])

  train_iter = IWSLT2017(root='/home/chub/datasets',
                         split='train', language_pair=('en', 'nl'))
  tgt_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, tgt_tokenizer, 1), specials=specials)
  tgt_vocab.set_default_index(tgt_vocab['<unk>'])

  return src_vocab, src_tokenizer, tgt_vocab, tgt_tokenizer


def make_nlp_task(batch_size):
  src_vocab, src_tokenizer, tgt_vocab, tgt_tokenizer = get_vocabs_and_tokenizer()

  def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
      src_batch.append(text_to_tensor(src_sample.rstrip('\n'), src_vocab, src_tokenizer))
      tgt_batch.append(text_to_tensor(tgt_sample.rstrip('\n'), tgt_vocab, tgt_tokenizer))
    src_batch = pad_sequence(src_batch, padding_value=src_vocab['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab['<pad>'], batch_first=True)
    return src_batch, tgt_batch

  train_iter = IWSLT2017(root='/home/chub/datasets',
                         split='train', language_pair=('en', 'nl'))
  train_loader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
  criterion = nn.NLLLoss(ignore_index=src_vocab['<pad>'])
  model = make_model(len(src_vocab), len(tgt_vocab), 6)

  return train_loader, criterion, model


def main(nlp_task):
  batch_size = 32
  if nlp_task:
    train_loader, criterion, model = make_nlp_task(batch_size)
  else:
    nbatches = 20
    train_loader, criterion, model = make_silly_task(nbatches, batch_size)


  optimizer = torch.optim.Adam(model.parameters(),
                               lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

  epochs = 10
  losses = []

  try:
    for _ in range(epochs):
      for src_batch, tgt_batch in tqdm(train_loader):
        if nlp_task:
          tgt_input = tgt_batch[:, :-1]
          src_mask, tgt_mask = create_masks(src_batch, tgt_input,
                                            src_vocab['<pad>'], tgt_vocab['<pad>'])
        else:
          tgt_input = tgt_batch
          src_mask, tgt_mask = create_masks(src_batch, tgt_input, -1, -1)

        out = model(src_batch, tgt_input, src_mask, tgt_mask)

        if nlp_task:
          tgt_output = tgt_batch[:, 1:]
        else:
          tgt_output = tgt_batch

        loss = criterion(out.transpose(1, 2), tgt_output)

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

  torch.save(model, 'transformer.pt')


if __name__ == "__main__":
  main()

