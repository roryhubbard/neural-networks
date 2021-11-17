import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ref: https://nlp.seas.harvard.edu/2018/04/03/attention.html


def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
  attn_shape = (1, size, size)
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  return torch.from_numpy(subsequent_mask) == 0


def scaled_dot_product_attention(Q, K, V, dk):
  scores = Q @ K.transpose(-2, -1) / np.sqrt(dk)
  return F.softmax(scores, dim=-1) @ V


class Transformer(nn.module):

  def __init__(self, d_model, d_ff, h, vocab):
    """
    d_model = dimension of layers between modules
    d_ff = dimension of hidden layer in feed forward sublayer
    h = number of heads in mulit-head-attention
    """
    super().__init__()
    self.embeddings = Embeddings(vocab, d_model)
    self.encoder = Encoder(d_model, d_ff, h)
    self.decoder = Decoder(d_model, d_ff, h)

  def forward(self, x):
    pass


class Encoder(nn.module):

  def __init__(self, d_model, d_ff, h):
    super().__init__()
    self.layers = clones(EncoderLayer(d_model, d_ff, h), 6)

  def forward(self, x, mask):
    for l in self.layers:
      h = l(x, mask)
    return h


class EncoderLayer(nn.module):

  def __init__(self, d_model, d_ff, h):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model, h)
    self.norm1 = LayerNorm(d_model)
    self.ff = PositionwiseFeedForward(d_model, d_ff)
    self.norm2 = LayerNorm(d_model)

  def forward(self, x, mask):
    h = self.self_attention(x, x, x, mask)
    h = self.norm1(h)
    h = self.ff(h)
    h = self.norm2(h)
    return h


class Decoder(nn.module):

  def __init__(self, d_model, d_ff, h):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model, h)
    self.norm1 = LayerNorm(d_model)
    self.source_attention = MultiHeadAttention(d_model, h)
    self.norm2 = LayerNorm(d_model)
    self.ff = PositionwiseFeedForward(d_model, d_ff)
    self.norm3 = LayerNorm(d_model)

  def forward(self, s, x, self_mask, source_mask):
    h = self.self_attention(s, s, s, self_mask)
    h = self.norm1(h)
    h = self.source_attention(s, x, x, source_mask)
    h = self.norm3(h)
    h = self.ff(h)
    h = self.norm2(h)
    return h


class MultiHeadAttention(nn.module):

  def __init__(self, d_model, h):
    super().__init__()
    assert d_model % h == 0
    dk = d_model // h
    self.WO = nn.Parameter(torch.empty((d_model, d_model)))
    self.layers = clones(SingleHeadAttention(d_model, dk), h)
    self.reset_weights()

  def reset_weights(self):
    nn.init.xavier_uniform_(self.WO))

  def forward(self, Q, K, V):
    monolith_head = torch.cat([l(Q, K, V) for l in self.layers], dim=-1)
    return monolith_head @ self.WO


class SingleHeadAttention(nn.module):

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


  def __init__(self, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.h1 = nn.Linear(d_model, d_ff)
    self.h2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.h2(self.dropout(F.relu(self.h1(x))))


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
    x = x + Variable(self.pe[:, :x.size(1)], 
                     requires_grad=False)
    return self.dropout(x)

def main():
  d_model = 512
  d_ff = 2048
  h = 8
  return


if __name__ == "__main__":
  main()

