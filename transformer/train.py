from tqdm import tqdm
import spacy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IWSLT2017

from transformer import Transformer


def additive_subsequent_mask(sz):
  mask = torch.triu(torch.ones((sz, sz)).type(torch.bool), diagonal=1)
  return torch.zeros((sz, sz)).masked_fill(mask, float('-inf'))


def create_masks(src, tgt, src_pad_val, tgt_pad_val):
  batch_size, src_seq_len = src.shape
  tgt_seq_len = tgt.shape[1]

  src_mask = torch.zeros_like(src).unsqueeze(1).float()
  tgt_mask = additive_subsequent_mask(tgt_seq_len).unsqueeze(0).expand(batch_size, -1, -1)

  src_pad_mask = (src == src_pad_val).unsqueeze(1)
  tgt_pad_mask = (tgt == tgt_pad_val).unsqueeze(1).expand(-1, tgt_seq_len, -1)

  src_mask.masked_fill(src_pad_mask, float('-inf'))
  tgt_mask.masked_fill(tgt_pad_mask, float('-inf'))

  return src_mask, tgt_mask


def text_to_tensor(text, vocab, tokenizer):
  return torch.cat((torch.tensor([vocab['<bos>']]),
                   torch.tensor(vocab(tokenizer(text))),
                   torch.tensor([vocab['<eos>']])))


def yield_tokens(data_iter, tokenizer, sample_idx):
  for data_sample in data_iter:
    yield tokenizer(data_sample[sample_idx])


def main():
  src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  tgt_tokenizer = get_tokenizer('spacy', language='nl_core_news_sm')
  specials = ['<unk>', '<pad>', '<bos>', '<eos>']

  train_iter = IWSLT2017(root='/home/chubb/datasets',
                         split='train', language_pair=('en', 'nl'))
  src_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, src_tokenizer, 0), specials=specials)
  src_vocab.set_default_index(src_vocab['<unk>'])

  train_iter = IWSLT2017(root='/home/chubb/datasets',
                         split='train', language_pair=('en', 'nl'))
  tgt_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, tgt_tokenizer, 1), specials=specials)
  tgt_vocab.set_default_index(tgt_vocab['<unk>'])

  def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
      src_batch.append(text_to_tensor(src_sample.rstrip('\n'), src_vocab, src_tokenizer))
      tgt_batch.append(text_to_tensor(tgt_sample.rstrip('\n'), tgt_vocab, tgt_tokenizer))
    src_batch = pad_sequence(src_batch, padding_value=src_vocab['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab['<pad>'], batch_first=True)
    return src_batch, tgt_batch

#  train_iter, val_iter, test_iter = IWSLT2017(root='/home/chubb/datasets',
#                                              language_pair=('en', 'nl'))
#  val_loader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)
#  test_loader = DataLoader(test_iter, batch_size=batch_size, collate_fn=collate_fn)

  train_iter = IWSLT2017(root='/home/chubb/datasets',
                         split='train', language_pair=('en', 'nl'))
  bs = 32
  train_loader = DataLoader(train_iter, batch_size=bs, collate_fn=collate_fn)

  d_model = 512
  d_ff = 2048
  h = 8
  nel = 3
  ndl = 3
  dropout = 0.1

  transformer = Transformer(d_model, d_ff, h, nel, ndl,
                            len(src_vocab), len(tgt_vocab), dropout)
  # I do this within the class modules
#  for p in transformer.parameters():
#    if p.dim() > 1:
#      nn.init.xavier_uniform_(p)

  optimizer = torch.optim.Adam(transformer.parameters(),
                               lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
  criterion = nn.NLLLoss(ignore_index=src_vocab['<pad>'])

  losses = []

  try:
    for src_batch, tgt_batch in tqdm(train_loader):
      tgt_input = tgt_batch[:, :-1]
      src_mask, tgt_mask = create_masks(src_batch, tgt_input,
                                        src_vocab['<pad>'], tgt_vocab['<pad>'])
      out = transformer(src_batch, tgt_input, src_mask, tgt_mask)
      tgt_output = tgt_batch[:, 1:]
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

if __name__ == "__main__":
  main()

