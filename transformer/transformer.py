import copy
from tqdm import tqdm
import numpy as np
import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IWSLT2017


def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_dot_product_attention(Q, K, V, mask, dk):
  scores = Q @ K.transpose(-2, -1) / np.sqrt(dk) + mask
  return F.softmax(scores, dim=-1) @ V


class Transformer(nn.Module):

  def __init__(self, d_model, d_ff, h, src_vocab, tgt_vocab, dropout):
    """
    d_model = dimension of layers between modules
    d_ff = dimension of hidden layer in feed forward sublayer
    h = number of heads in mulit-head-attention
    """
    super().__init__()
    self.src_embed = Embed(src_vocab, d_model, dropout)
    self.tgt_embed = Embed(tgt_vocab, d_model, dropout)
    self.encoder = Encoder(d_model, d_ff, h, dropout)
    self.decoder = Decoder(d_model, d_ff, h, dropout)
    self.generator = Generator(d_model, tgt_vocab)

  def forward(self, src, tgt, src_mask, tgt_mask):
    h = self.encode(src, src_mask)
    h = self.decode(h, tgt, src_mask, tgt_mask)
    return self.generator(h)

  def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)

  def decode(self, memory, tgt, src_mask, tgt_mask):
    return self.decoder(memory, self.tgt_embed(tgt), src_mask, tgt_mask)


class Generator(nn.Module):

  def __init__(self, d_model, vocab):
    super().__init__()
    self.proj = nn.Linear(d_model, vocab)

  def forward(self, x):
    return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):

  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()
    self.layers = clones(EncoderLayer(d_model, d_ff, h, dropout), 6)

  def forward(self, x, mask):
    for l in self.layers:
      x = l(x, mask)
    return x


class EncoderLayer(nn.Module):

  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()
    self.src_attn = MultiHeadAttention(d_model, h, dropout)
    self.norm1 = LayerNorm(d_model)
    self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.norm2 = LayerNorm(d_model)

  def forward(self, x, mask):
    h = self.src_attn(x, x, x, mask)
    h = self.norm1(h)
    h = self.ff(h)
    return self.norm2(h)


class Decoder(nn.Module):

  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()
    self.layers = clones(DecoderLayer(d_model, d_ff, h, dropout), 6)

  def forward(self, memory, tgt, src_mask, tgt_mask):
    for l in self.layers:
      tgt = l(memory, tgt, src_mask, tgt_mask)
    return tgt


class DecoderLayer(nn.Module):

  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()
    self.tgt_attn = MultiHeadAttention(d_model, h, dropout)
    self.norm1 = LayerNorm(d_model)
    self.src_attn = MultiHeadAttention(d_model, h, dropout)
    self.norm2 = LayerNorm(d_model)
    self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.norm3 = LayerNorm(d_model)

  def forward(self, memory, tgt, src_mask, tgt_mask):
    h = self.tgt_attn(tgt, tgt, tgt, tgt_mask)
    h = self.norm1(h)
    h = self.src_attn(tgt, memory, memory, src_mask)
    h = self.norm3(h)
    h = self.ff(h)
    return self.norm2(h)


class MultiHeadAttention(nn.Module):

  def __init__(self, d_model, h, dropout):
    super().__init__()
    assert d_model % h == 0
    dk = d_model // h
    self.WO = nn.Parameter(torch.empty((d_model, d_model)))
    self.layers = clones(SingleHeadAttention(d_model, dk), h)
    self.dropout = nn.Dropout(dropout)
    self.reset_weights()

  def reset_weights(self):
    nn.init.xavier_uniform_(self.WO)

  def forward(self, Q, K, V, mask):
    monolith_head = torch.cat([l(Q, K, V, mask) for l in self.layers], dim=-1)
    h = monolith_head @ self.WO
    return self.dropout(h)


class SingleHeadAttention(nn.Module):

  def __init__(self, d_model, dk):
    """
    dk = dimension of projected subspace for query, key, and value
      - Assumes dk == dv (dimension of projected value subspace can be different)
    """
    super().__init__()
    self.dk = dk
    self.WQ = nn.Parameter(torch.empty((d_model, dk)))
    self.WK = nn.Parameter(torch.empty((d_model, dk)))
    self.WV = nn.Parameter(torch.empty((d_model, dk)))
    self.reset_weights()

  def reset_weights(self):
    nn.init.xavier_uniform_(self.WQ)
    nn.init.xavier_uniform_(self.WK)
    nn.init.xavier_uniform_(self.WV)

  def forward(self, Q, K, V, mask):
    query = Q @ self.WQ
    key = K @ self.WK
    value = V @ self.WV
    return scaled_dot_product_attention(query, key, value, mask, self.dk)


class PositionwiseFeedForward(nn.Module):

  def __init__(self, d_model, d_ff, dropout):
    super().__init__()
    self.h1 = nn.Linear(d_model, d_ff)
    self.h2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    h = self.h1(x)
    h = F.relu(h)
    h = self.h2(h)
    return self.dropout(h)


class LayerNorm(nn.Module):

  def __init__(self, size):
    super().__init__()
    self.a = nn.Parameter(torch.ones(size))
    self.b = nn.Parameter(torch.zeros(size))

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.a * (x - mean) / (std + 1e-6) + self.b


class Embed(nn.Module):

  def __init__(self, vocab, d_model, dropout):
    super().__init__()
    self.embedding = nn.Embedding(vocab, d_model)
    self.positional_encoding = PositionalEncoding(d_model, dropout)
    self.d_model = d_model

  def forward(self, x):
    h = self.embedding(x) * np.sqrt(self.d_model)
    return self.positional_encoding(h)


#class PositionalEncoding(nn.Module):
#
#  def __init__(self, d_model, dropout, max_len=5000):
#    super().__init__()
#    self.dropout = nn.Dropout(p=dropout)
#    pe = torch.zeros(max_len, d_model)
#    position = torch.arange(0, max_len).unsqueeze(1)
#    div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
#    pe[:, 0::2] = torch.sin(position * div_term)
#    pe[:, 1::2] = torch.cos(position * div_term)
#    pe = pe.unsqueeze(0)
#    self.register_buffer('pe', pe)
#
#  def forward(self, x):
#    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#    return self.dropout(x)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, maxlen=5000):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    pos_embedding = torch.zeros((maxlen, d_model))
    pos = torch.arange(0, maxlen).unsqueeze(1)
    den = torch.exp(-torch.arange(0, d_model, 2) * np.log(10000) / d_model)
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    pos_embedding = pos_embedding.unsqueeze(-2)

    self.register_buffer('pos_embedding', pos_embedding)

  def forward(self, x):
    h = x + self.pos_embedding[:x.size(0), :]
    return self.dropout(h)


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
  dropout = 0.1

  transformer = Transformer(d_model, d_ff, h,
                            len(src_vocab), len(tgt_vocab), dropout)
  optimizer = torch.optim.Adam(transformer.parameters(),
                               lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
  criterion = nn.NLLLoss(ignore_index=src_vocab['<pad>'])

  for src_batch, tgt_batch in tqdm(train_loader):
    src_mask, tgt_mask = create_masks(src_batch, tgt_batch,
                                      src_vocab['<pad>'], tgt_vocab['<pad>'])
    out = transformer(src_batch, tgt_batch, src_mask, tgt_mask)
    loss = criterion(out.transpose(1, 2), tgt_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
  main()

