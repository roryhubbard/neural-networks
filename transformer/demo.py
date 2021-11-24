import torch
from utils import additive_subsequent_mask
from train_nlp import make_nlp_task, text_to_tensor


def greedy_decode(model, text, src_vocab, src_tokenizer, tgt_vocab, maxlen=50):
  src = text_to_tensor(text.rstrip('\n'), src_vocab, src_tokenizer).unsqueeze(0)
  memory = model.encode(src)
  last_word = tgt_vocab['<bos>']
  running_output = torch.tensor([last_word]).unsqueeze(0)
  i = 0

  while last_word != tgt_vocab['<eos>'] and i < maxlen:
    tgt_mask = additive_subsequent_mask(running_output.size(1))
    out = model.decode(memory, running_output, tgt_mask)
    prob = model.generator(out[:, -1])
    last_output = torch.argmax(prob, dim=-1).unsqueeze(0)
    running_output = torch.cat([running_output, last_output], dim=1)
    last_word = last_output.item()
    i += 1

  out_text = tgt_vocab.lookup_tokens(running_output.flatten().tolist())
  return ' '.join(out_text)


def main(trained_model_path):
  src_vocab, src_tokenizer, tgt_vocab, tgt_tokenizer = make_nlp_task(1, split='test')
  model = torch.load(trained_model_path).eval()

  for en_text, nl_text in train_iter:
    print('ENGLISH PROMPT\n', en_text.rstrip('\n'))
    print('EXPECTED TRANSLATION\n', nl_text.rstrip('\n'))

    decoded = greedy_decode(model, en_text, src_vocab, src_tokenizer, tgt_vocab)

    print('MODEL OUTPUT\n', decoded, '\n')


if __name__ == "__main__":
  trained_model_path = 'en2nl_transformer.pt'
  main(trained_model_path)

