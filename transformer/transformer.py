import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_dot_product_attention(Q, K, V, mask, dk):
  scores = Q @ K.transpose(-2, -1) / np.sqrt(dk) + mask
  return F.softmax(scores, dim=-1) @ V


class Transformer(nn.Module):

  def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
               num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.encoder = Encoder(num_encoder_layers, d_model, dim_feedforward, nhead, dropout)
    self.decoder = Decoder(num_decoder_layers, d_model, dim_feedforward, nhead, dropout)

#  def reset_weights(self):
#    nn.init.xavier_uniform_(self.src_embed.embedding.weight)
#    self.src_embed.embedding.weight = \
#      self.tgt_embed.embedding.weight = self.generator.fc.weight

  def forward(self, src, tgt, src_mask=0, tgt_mask=0):
    h = self.encode(src, src_mask)
    h = self.decode(tgt, h, src_mask, tgt_mask)
    return h

  def encode(self, src, src_mask=0):
    return self.encoder(src, src_mask)

  def decode(self, tgt, memory, src_mask=0, tgt_mask=0):
    return self.decoder(tgt, memory, src_mask, tgt_mask)


class Generator(nn.Module):

  def __init__(self, d_model, vocab):
    super().__init__()
    self.proj = nn.Linear(d_model, vocab)

  def forward(self, x):
    h = self.proj(x)
    return F.log_softmax(h, dim=-1)


class Encoder(nn.Module):

  def __init__(self, num_encoder_layers, d_model, dim_feedforward, nhead, dropout):
    super().__init__()
    self.layers = clones(
      EncoderLayer(d_model, dim_feedforward, nhead, dropout), num_encoder_layers)

  def forward(self, x, mask):
    for l in self.layers:
      x = l(x, mask)
    return x


class EncoderLayer(nn.Module):

  def __init__(self, d_model, dim_feedforward, nhead, dropout):
    super().__init__()
    self.src_attn = MultiHeadAttention(d_model, nhead, dropout)
    self.norm1 = LayerNorm(d_model)
    self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
    self.norm2 = LayerNorm(d_model)

  def forward(self, x, mask):
    h = self.src_attn(x, x, x, mask)
    h = self.norm1(h)
    h = self.ff(h)
    return self.norm2(h)


class Decoder(nn.Module):

  def __init__(self, num_decoder_layers, d_model, dim_feedforward, nhead, dropout):
    super().__init__()
    self.layers = clones(
      DecoderLayer(d_model, dim_feedforward, nhead, dropout), num_decoder_layers)

  def forward(self, tgt, memory, src_mask, tgt_mask):
    for l in self.layers:
      tgt = l(tgt, memory, src_mask, tgt_mask)
    return tgt


class DecoderLayer(nn.Module):

  def __init__(self, d_model, dim_feedforward, nhead, dropout):
    super().__init__()
    self.tgt_attn = MultiHeadAttention(d_model, nhead, dropout)
    self.norm1 = LayerNorm(d_model)
    self.src_attn = MultiHeadAttention(d_model, nhead, dropout)
    self.norm2 = LayerNorm(d_model)
    self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
    self.norm3 = LayerNorm(d_model)

  def forward(self, tgt, memory, src_mask, tgt_mask):
    h = self.tgt_attn(tgt, tgt, tgt, tgt_mask)
    h = self.norm1(h)
    h = self.src_attn(tgt, memory, memory, src_mask)
    h = self.norm3(h)
    h = self.ff(h)
    return self.norm2(h)


class MultiHeadAttention(nn.Module):

  def __init__(self, d_model, nhead, dropout):
    super().__init__()
    assert d_model % nhead == 0
    dk = d_model // nhead
    self.WO = nn.Parameter(torch.empty((d_model, d_model)))
    self.layers = clones(SingleHeadAttention(d_model, dk), nhead)
    self.dropout = nn.Dropout(dropout)

  def forward(self, Q, K, V, mask):
    monolith_head = torch.cat([l(Q, K, V, mask) for l in self.layers], dim=-1)
    h = monolith_head @ self.WO
    return self.dropout(h)


class SingleHeadAttention(nn.Module):

  def __init__(self, d_model, dk):
    super().__init__()
    self.dk = dk
    self.WQ = nn.Parameter(torch.empty((d_model, dk)))
    self.WK = nn.Parameter(torch.empty((d_model, dk)))
    self.WV = nn.Parameter(torch.empty((d_model, dk)))

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


class TokenEmbedding(nn.Module):

  def __init__(self, vocab, d_model):
    super().__init__()
    self.embedding = nn.Embedding(vocab, d_model)
    self.d_model = d_model

  def forward(self, x):
    return self.embedding(x) * np.sqrt(self.d_model)


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
      x = x + Variable(self.pe[:, :x.size(1)], 
                       requires_grad=False)
      return self.dropout(x)

