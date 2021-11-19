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


def scaled_dot_product_attention(Q, K, V, dk):
  scores = Q @ K.transpose(-2, -1) / np.sqrt(dk)
  return F.softmax(scores, dim=-1) @ V


class Transformer(nn.Module):

  def __init__(self, d_model, d_ff, h, src_vocab, tgt_vocab, dropout):
    """
    d_model = dimension of layers between modules
    d_ff = dimension of hidden layer in feed forward sublayer
    h = number of heads in mulit-head-attention
    """
    super().__init__()
    self.embeddings = Embeddings(src_vocab, d_model)
    self.positional_encoder = PositionalEncoding(d_model, dropout)
    self.encoder = Encoder(d_model, d_ff, h, dropout)
    self.decoder = Decoder(d_model, d_ff, h, dropout)
    self.generator = Generator(d_model, tgt_vocab)

  def forward(self, x):
    h = self.embeddings(x)
    h = self.position_encoder(h)
    h = self.encoder(h)
    h = self.decoder(h)
    return self.generator(x)


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
      h = l(x, mask)
    return h


class EncoderLayer(nn.Module):

  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model, h, dropout)
    self.norm1 = LayerNorm(d_model)
    self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.norm2 = LayerNorm(d_model)

  def forward(self, x, mask):
    h = self.self_attention(x, x, x, mask)
    h = self.norm1(h)
    h = self.ff(h)
    h = self.norm2(h)
    return h


class Decoder(nn.Module):

  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()
    self.layers = clones(DecoderLayer(d_model, d_ff, h, dropout), 6)

  def forward(self, x, mask):
    for l in self.layers:
      h = l(x, mask)
    return h


class DecoderLayer(nn.Module):

  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model, h, dropout)
    self.norm1 = LayerNorm(d_model)
    self.source_attention = MultiHeadAttention(d_model, h, dropout)
    self.norm2 = LayerNorm(d_model)
    self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    self.norm3 = LayerNorm(d_model)

  def forward(self, s, x, self_mask, source_mask):
    h = self.self_attention(s, s, s, self_mask)
    h = self.norm1(h)
    h = self.source_attention(s, x, x, source_mask)
    h = self.norm3(h)
    h = self.ff(h)
    h = self.norm2(h)
    return h


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

  def forward(self, Q, K, V):
    monolith_head = torch.cat([l(Q, K, V) for l in self.layers], dim=-1)
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

  def forward(self, Q, K, V):
    query = Q @ self.WQ
    key = K @ self.WK
    value = V @ self.WV
    return scaled_dot_product_attention(query, key, value, self.dk)


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


class Embeddings(nn.Module):

  def __init__(self, vocab, d_model):
    super().__init__()
    self.embed = nn.Embedding(vocab, d_model)
    self.d_model = d_model

  def forward(self, x):
    return self.embed(x) * np.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout, max_len=5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    return self.dropout(x)


def additive_subsequent_mask(sz):
  mask = torch.triu(torch.ones((sz, sz)).type(torch.bool), diagonal=1)
  return torch.zeros((sz, sz)).masked_fill(mask, float('-inf'))


def create_mask(src, tgt):
  src_seq_len = src.shape[0]
  tgt_seq_len = tgt.shape[0]

  tgt_mask = additive_subsequent_mask(tgt_seq_len)
  src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

  src_padding_mask = (src == PAD_IDX).transpose(0, 1)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def text_to_tensor(text, vocab, tokenizer):
  return torch.cat((torch.tensor([vocab['<bos>']]),
                   torch.tensor(vocab(tokenizer(text))),
                   torch.tensor([vocab['<eos>']])))


def main():
  src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  tgt_tokenizer = get_tokenizer('spacy', language='nl_core_news_sm')

  train_iter = IWSLT2017(root='/home/chubb/datasets',
                         split='train', language_pair=('en', 'nl'))

  src_train, tgt_train = list(zip(*list(train_iter)))

  specials = ['<unk>', '<pad>', '<bos>', '<eos>']
  src_vocab = build_vocab_from_iterator(map(src_tokenizer, src_train), specials=specials)
  src_vocab.set_default_index(src_vocab['<unk>'])
  tgt_vocab = build_vocab_from_iterator(map(tgt_tokenizer, tgt_train), specials=specials)
  tgt_vocab.set_default_index(tgt_vocab['<unk>'])

  def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
      src_batch.append(text_to_tensor(src_sample.rstrip('\n'), src_vocab, src_tokenizer))
      tgt_batch.append(text_to_tensor(tgt_sample.rstrip('\n'), tgt_vocab, tgt_tokenizer))

    src_batch = pad_sequence(src_batch, padding_value=src_vocab['<pad>'])
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab['<pad>'])
    return src_batch, tgt_batch

  train_iter, val_iter, test_iter = IWSLT2017(root='/home/chubb/datasets',
                                              language_pair=('en', 'nl'))

  batch_size = 64
  train_loader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
  val_loader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)
  test_loader = DataLoader(test_iter, batch_size=batch_size, collate_fn=collate_fn)

  d_model = 512
  d_ff = 2048
  h = 8
  dropout = 0.1

  transformer = Transformer(d_model, d_ff, h, len(src_vocab), len(tgt_vocab), dropout)

  for src_batch, tgt_batch in tqdm(train_loader):
    print(src_batch.shape)
    print(tgt_batch.shape)
    return


if __name__ == "__main__":
  main()

