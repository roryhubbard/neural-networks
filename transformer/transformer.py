import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_dot_product_attention(Q, K, V, mask, dk):
  scores = Q @ K.transpose(-2, -1) / np.sqrt(dk) + mask
  return F.softmax(scores, dim=-1) @ V


class Transformer(nn.Module):

  def __init__(self, d_model, d_ff, h, nel, ndl, src_vocab, tgt_vocab, dropout):
    """
    d_model = dimension of layers between modules
    d_ff = dimension of hidden layer in feed forward sublayer
    h = number of heads in mulit-head-attention
    """
    super().__init__()
    self.src_embed = Embed(src_vocab, d_model, dropout)
    self.tgt_embed = Embed(tgt_vocab, d_model, dropout)
    self.encoder = Encoder(nel, d_model, d_ff, h, dropout)
    self.decoder = Decoder(ndl, d_model, d_ff, h, dropout)
    self.generator = Generator(d_model, tgt_vocab)
    self.reset_weights()

  def reset_weights(self):
    nn.init.xavier_uniform_(self.src_embed.embedding.weight)
    self.src_embed.embedding.weight = \
      self.tgt_embed.embedding.weight = self.generator.fc.weight

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
    self.fc = nn.Linear(d_model, vocab)

  def forward(self, x):
    return F.log_softmax(self.fc(x), dim=-1)


class Encoder(nn.Module):

  def __init__(self, nel, d_model, d_ff, h, dropout):
    super().__init__()
    self.layers = clones(EncoderLayer(d_model, d_ff, h, dropout), nel)

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

  def __init__(self, ndl, d_model, d_ff, h, dropout):
    super().__init__()
    self.layers = clones(DecoderLayer(d_model, d_ff, h, dropout), ndl)

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

