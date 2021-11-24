import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformer import Transformer


class FullModel(nn.Module):

  def __init__(self, transformer, src_tok_embed, tgt_tok_embed, pos_encod, generator):
    super().__init__()
    self.transformer = transformer
    self.src_tok_embed = src_tok_embed
    self.tgt_tok_embed = tgt_tok_embed
    self.pos_encod = pos_encod
    self.generator = generator

  def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
              src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
    src_emb = self.pos_encod(self.src_tok_embed(src))
    tgt_emb = self.pos_encod(self.tgt_tok_embed(tgt))
    out = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, memory_mask,
                           src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
    return self.generator(out)

  def encode(self, src, src_mask=None, src_key_padding_mask=None):
    h = self.pos_encod(self.src_tok_embed(src))
    return self.transformer.encoder(h, src_mask, src_key_padding_mask)

  def decode(self, tgt, memory, tgt_mask=None, memory_mask=None,
             tgt_key_padding_mask=None, memory_key_padding_mask=None):
    h = self.pos_encod(self.tgt_tok_embed(tgt))
    return self.transformer.decoder(h, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask)


class Generator(nn.Module):

  def __init__(self, d_model, vocab):
    super().__init__()
    self.proj = nn.Linear(d_model, vocab)

  def forward(self, x):
    h = self.proj(x)
    return F.log_softmax(h, dim=-1)


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
    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    return self.dropout(x)


def additive_subsequent_mask(sz):
  mask = torch.triu(torch.ones((sz, sz)).type(torch.bool), diagonal=1)
  return torch.zeros((sz, sz)).masked_fill(mask, float('-inf'))


def create_masks(src, tgt, src_pad_val, tgt_pad_val):
  batch_size, src_seq_len = src.shape
  tgt_seq_len = tgt.shape[1]

  src_mask = torch.zeros((src_seq_len, src_seq_len)).float()
  tgt_mask = additive_subsequent_mask(tgt_seq_len)

  # TODO: create these arrays these
  memory_mask = src_key_padding_mask = tgt_key_padding_mask = memory_key_padding_mask = None
#  src_pad_mask = (src == src_pad_val).unsqueeze(1)
#  tgt_pad_mask = (tgt == tgt_pad_val).unsqueeze(1).expand(-1, tgt_seq_len, -1)
#  src_mask.masked_fill(src_pad_mask, float('-inf'))
#  tgt_mask.masked_fill(tgt_pad_mask, float('-inf'))

  return src_mask, tgt_mask, memory_mask, \
      src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask


def make_model(src_vocab_size, tgt_vocab_size, N=6, custom_transformer=False):
  dropout = 0.1
  if custom_transformer:
    transformer = Transformer(num_encoder_layers=N,
                              num_decoder_layers=N, dropout=dropout)
  else:
    transformer = nn.Transformer(num_encoder_layers=N, num_decoder_layers=N,
                                 dropout=dropout, batch_first=True)

  src_embedding = TokenEmbedding(src_vocab_size, transformer.d_model)
  tgt_embedding = TokenEmbedding(tgt_vocab_size, transformer.d_model)
  positional_encoding = PositionalEncoding(transformer.d_model, dropout)
  generator = Generator(transformer.d_model, tgt_vocab_size)

  model = FullModel(transformer, src_embedding, tgt_embedding, positional_encoding, generator)

  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return model

