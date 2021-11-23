import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_dot_product_attention(Q, K, V, dk, additive_mask=None, key_padding_mask=None):
  scores = Q @ K.transpose(-2, -1) / np.sqrt(dk)
  if additive_mask is not None:
    # unsqueeze batch dimension can be broadcasted
    additive_mask = additive_mask.unsqueeze(0)
    scores += additive_mask
  if key_padding_mask is not None:
    pass
  return F.softmax(scores, dim=-1) @ V


class Transformer(nn.Module):

  def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
               num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.encoder = Encoder(num_encoder_layers, d_model, dim_feedforward, nhead, dropout)
    self.decoder = Decoder(num_decoder_layers, d_model, dim_feedforward, nhead, dropout)

  def forward(self, src, tgt, src_mask=None, tgt_mask=None,
              memory_mask=None, src_key_padding_mask=None,
              tgt_key_padding_mask=None, memory_key_padding_mask=None):
    h = self.encode(src, src_mask, src_key_padding_mask)
    return self.decode(tgt, h, tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask)

  def encode(self, src, src_mask=None, src_key_padding_mask=None):
    return self.encoder(src, src_mask, src_key_padding_mask)

  def decode(self, tgt, memory, tgt_mask=None, memory_mask=None,
             tgt_key_padding_mask=None, memory_key_padding_mask=None):
    return self.decoder(tgt, memory, tgt_mask, memory_mask,
                        tgt_key_padding_mask, memory_key_padding_mask)


class ResidualDropoutNormalize(nn.Module):

  def __init__(self, d_model, dropout):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNorm(d_model)

  def forward(self, x, sublayer):
    h = x + self.dropout(sublayer(x))
    return self.norm(h)


class Encoder(nn.Module):

  def __init__(self, num_encoder_layers, d_model, dim_feedforward, nhead, dropout):
    super().__init__()
    self.layers = clones(
      EncoderLayer(d_model, dim_feedforward, nhead, dropout), num_encoder_layers)

  def forward(self, x, src_mask, src_key_padding_mask):
    for l in self.layers:
      x = l(x, src_mask, src_key_padding_mask)
    return x


class EncoderLayer(nn.Module):

  def __init__(self, d_model, dim_feedforward, nhead, dropout):
    super().__init__()
    self.src_attn = MultiHeadAttention(d_model, nhead)
    self.sublayer1 = ResidualDropoutNormalize(d_model, dropout)
    self.ff = PositionwiseFeedForward(d_model, dim_feedforward)
    self.sublayer2 = ResidualDropoutNormalize(d_model, dropout)

  def forward(self, x, src_mask, src_key_padding_mask):
    sublayer = lambda x: self.src_attn(x, x, x, src_mask, src_key_padding_mask)
    h = self.sublayer1(x, sublayer)
    return self.sublayer2(h, self.ff)


class Decoder(nn.Module):

  def __init__(self, num_decoder_layers, d_model, dim_feedforward, nhead, dropout):
    super().__init__()
    self.layers = clones(
      DecoderLayer(d_model, dim_feedforward, nhead, dropout), num_decoder_layers)

  def forward(self, tgt, memory, tgt_mask, memory_mask,
              tgt_key_padding_mask, memory_key_padding_mask):
    for l in self.layers:
      tgt = l(tgt, memory, tgt_mask, memory_mask,
              tgt_key_padding_mask, memory_key_padding_mask)
    return tgt


class DecoderLayer(nn.Module):

  def __init__(self, d_model, dim_feedforward, nhead, dropout):
    super().__init__()
    self.tgt_attn = MultiHeadAttention(d_model, nhead)
    self.sublayer1 = ResidualDropoutNormalize(d_model, dropout)
    self.memory_attn = MultiHeadAttention(d_model, nhead)
    self.sublayer2 = ResidualDropoutNormalize(d_model, dropout)
    self.ff = PositionwiseFeedForward(d_model, dim_feedforward)
    self.sublayer3 = ResidualDropoutNormalize(d_model, dropout)

  def forward(self, tgt, memory, tgt_mask, memory_mask,
              tgt_key_padding_mask, memory_key_padding_mask):
    sublayer = lambda tgt: self.tgt_attn(tgt, tgt, tgt, tgt_mask, tgt_key_padding_mask)
    h = self.sublayer1(tgt, sublayer)
    sublayer = lambda h: self.memory_attn(h, memory, memory, memory_mask, memory_key_padding_mask)
    h = self.sublayer2(h, sublayer)
    return self.sublayer3(h, self.ff)


class MultiHeadAttention(nn.Module):

  def __init__(self, d_model, nhead):
    super().__init__()
    assert d_model % nhead == 0
    dk = d_model // nhead
    self.WO = nn.Parameter(torch.empty((d_model, d_model)))
    self.layers = clones(SingleHeadAttention(d_model, dk), nhead)

  def forward(self, Q, K, V, additive_mask, key_padding_mask):
    monolith_head = torch.cat([l(Q, K, V, additive_mask, key_padding_mask)
                               for l in self.layers], dim=-1)
    return monolith_head @ self.WO


class SingleHeadAttention(nn.Module):

  def __init__(self, d_model, dk):
    super().__init__()
    self.dk = dk
    self.WQ = nn.Parameter(torch.empty((d_model, dk)))
    self.WK = nn.Parameter(torch.empty((d_model, dk)))
    self.WV = nn.Parameter(torch.empty((d_model, dk)))

  def forward(self, Q, K, V, additive_mask, key_padding_mask):
    query = Q @ self.WQ
    key = K @ self.WK
    value = V @ self.WV
    return scaled_dot_product_attention(query, key, value, self.dk,
                                        additive_mask, key_padding_mask)


class PositionwiseFeedForward(nn.Module):

  def __init__(self, d_model, d_ff):
    super().__init__()
    self.h1 = nn.Linear(d_model, d_ff)
    self.h2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    h = self.h1(x)
    h = F.relu(h)
    return self.h2(h)


class LayerNorm(nn.Module):

  def __init__(self, size):
    super().__init__()
    self.a = nn.Parameter(torch.ones(size))
    self.b = nn.Parameter(torch.zeros(size))

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.a * (x - mean) / (std + 1e-6) + self.b

