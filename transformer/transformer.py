import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.module):

  def __init__(self, d_model=512):
    super().__init__()
    self.encoder = Encoder(d_model)
    self.decoder = Decoder(d_model)

  def forward(self, x):
    pass


class Encoder(nn.module):

  def __init__(self, d_model):
    super().__init__()

  def forward(self, x):
    pass


class EncoderLayer(nn.module):

  def __init__(self, d_model):
    super().__init__()

  def forward(self, x):
    pass


class Decoder(nn.module):

  def __init__(self, d_model):
    super().__init__()

  def forward(self, x):
    pass


class Sublayer(nn.module):

  def __init__(self, d_model, layer_type):
    self.function = MultiHeadAttention(d_model) if layer_type == 'multi-head-attention' \
      else FeedForward(d_model)
    self.norm = LayerNorm(d_model)

  def forward(self, x):
    return self.norm(x + self.function(x))


def attention(Q, K, V):
  dk = Q.size(-1)
  return F.softmax(Q @ K.T / math.sqrt(dk)) @ V


class MultiHeadAttention(nn.module):

  def __init__(self, d_model, h):
    super().__init__()
    assert d_model % h == 0
    self.a = nn.Parameter(torch.ones(size))

  def forward(self, x):
    pass


class SingleHeadAttention(nn.module):

  def __init__(self, d_model, dk):
    super().__init__()
    self.WQ = nn.Parameter(torch.empty((d_model, dk)))
    self.WK = nn.Parameter(torch.empty((d_model, dk)))
    self.WV = nn.Parameter(torch.empty((d_model, dk)))

  def reset_weights(self):
    nn.init.xavier_uniform_(self.WQ)
    nn.init.xavier_uniform_(self.WK)
    nn.init.xavier_uniform_(self.WV)

  def forward(self, x):
    pass


class FeedForward(nn.module):

  def __init(self):
    super().__init__()

  def forward(self, x):
    pass


class LayerNorm(nn.Module):

  def __init__(self, size):
    super().__init__()
    self.a = nn.Parameter(torch.ones(size))
    self.b = nn.Parameter(torch.zeros(size))

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.a * (x - mean) / (std + 1e-6) + self.b


def main():
  pass
#  for p in model.parameters():
#    if p.dim() > 1:
#      nn.init.xavier_uniform(p)


if __name__ == "__main__":
  main()
