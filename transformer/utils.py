import torch
import torch.nn as nn

from transformer import Transformer, TokenEmbedding, PositionalEncoding, Generator


class FullModel(nn.Module):

  def __init__(self, transformer, src_tok_embed, tgt_tok_embed, pos_encod, generator):
    super().__init__()
    self.transformer = transformer
    self.src_tok_embed = src_tok_embed
    self.tgt_tok_embed = tgt_tok_embed
    self.pos_encod = pos_encod
    self.generator = generator

  def forward(self, src, tgt, src_mask, tgt_mask):
    src_emb = self.pos_encod(self.src_tok_embed(src))
    tgt_emb = self.pos_encod(self.tgt_tok_embed(tgt))
    out = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
    return self.generator(out)

  def encode(self, src, src_mask):
    h = self.pos_encod(self.src_tok_embed(src))
    return self.transformer.encoder(h, src_mask)

  def decode(self, tgt, memory, tgt_mask):
    h = self.pos_encod(self.tgt_tok_embed(tgt))
    return self.transformer.decoder(h, memory, tgt_mask)


def additive_subsequent_mask(sz):
  mask = torch.triu(torch.ones((sz, sz)).type(torch.bool), diagonal=1)
  return torch.zeros((sz, sz)).masked_fill(mask, float('-inf'))


def create_masks(src, tgt, src_pad_val, tgt_pad_val):
  batch_size, src_seq_len = src.shape
  tgt_seq_len = tgt.shape[1]

  src_mask = torch.zeros((src_seq_len, src_seq_len)).float()
  tgt_mask = additive_subsequent_mask(tgt_seq_len)

#  src_pad_mask = (src == src_pad_val).unsqueeze(1)
#  tgt_pad_mask = (tgt == tgt_pad_val).unsqueeze(1).expand(-1, tgt_seq_len, -1)
#
#  src_mask.masked_fill(src_pad_mask, float('-inf'))
#  tgt_mask.masked_fill(tgt_pad_mask, float('-inf'))

  return src_mask, tgt_mask


def make_model(src_vocab_size, tgt_vocab_size, N=6, custom_transformer=False):
  if custom_transformer:
    transformer = Transformer(num_encoder_layers=N, num_decoder_layers=N)
  else:
    transformer = nn.Transformer(num_encoder_layers=N,
                                 num_decoder_layers=N, batch_first=True)

  src_embedding = TokenEmbedding(src_vocab_size, transformer.d_model)
  tgt_embedding = TokenEmbedding(tgt_vocab_size, transformer.d_model)
  positional_encoding = PositionalEncoding(transformer.d_model, 0.1)
  generator = Generator(transformer.d_model, tgt_vocab_size)

  model = FullModel(transformer, src_embedding, tgt_embedding, positional_encoding, generator)

  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return model

