import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IWSLT2017


def additive_subsequent_mask(sz):
  mask = torch.triu(torch.ones((sz, sz)).type(torch.bool), diagonal=1)
  return torch.zeros((sz, sz)).masked_fill(mask, float('-inf'))


def yield_tokens(data_iter, tokenizer, sample_idx):
  for data_sample in data_iter:
    yield tokenizer(data_sample[sample_idx])


def text_to_tensor(text, vocab, tokenizer):
  return torch.cat((torch.tensor([vocab['<bos>']]),
                   torch.tensor(vocab(tokenizer(text))),
                   torch.tensor([vocab['<eos>']])))


def greedy_decode(model, text, src_vocab, src_tokenizer, tgt_vocab, maxlen=100):
  src = text_to_tensor(text.rstrip('\n'), src_vocab, src_tokenizer).unsqueeze(0)
  memory = model.encode(src)
  last_word = tgt_vocab['<bos>']
  running_output = torch.tensor([last_word]).unsqueeze(0)
  i = 0
  while last_word != tgt_vocab['<eos>'] and i < maxlen:
    tgt_mask = additive_subsequent_mask(running_output.size(1))
    out = model.decode(memory, running_output, 0, tgt_mask)
    prob = model.generator(out[:, -1])
    last_output = torch.argmax(prob, dim=-1).unsqueeze(0)
    running_output = torch.cat([running_output, last_output], dim=1)
    last_word = last_output.item()
    i += 1
  out_text = tgt_vocab.lookup_tokens(running_output.flatten().tolist())
  return out_text


def main():
  src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  tgt_tokenizer = get_tokenizer('spacy', language='nl_core_news_sm')
  specials = ['<unk>', '<pad>', '<bos>', '<eos>']

  train_iter = IWSLT2017(root='/home/chub/datasets',
                         split='train', language_pair=('en', 'nl'))
  src_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, src_tokenizer, 0), specials=specials)
  src_vocab.set_default_index(src_vocab['<unk>'])

  train_iter = IWSLT2017(root='/home/chub/datasets',
                         split='train', language_pair=('en', 'nl'))
  tgt_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, tgt_tokenizer, 1), specials=specials)
  tgt_vocab.set_default_index(tgt_vocab['<unk>'])

  train_iter = IWSLT2017(root='/home/chub/datasets',
                         split='train', language_pair=('en', 'nl'))

  transformer = torch.load('trained_transformer.pt').eval()
  for en_text, nl_text in train_iter:
    print('ENGLISH PROMPT\n', en_text.rstrip('\n'))
    print('EXPECTED TRANSLATION\n', nl_text.rstrip('\n'))
    decoded = greedy_decode(transformer, en_text, src_vocab, src_tokenizer, tgt_vocab)
    print('MODEL OUTPUT\n', decoded)

    break


if __name__ == "__main__":
  main()

