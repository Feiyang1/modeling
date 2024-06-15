from data_loader import get_data
from trainer import Trainer
from params import Word2VecParams
from torchtext.data.utils import get_tokenizer
from vocab import build_vocab
from preprocessing import SkipGrams
from model import Model
import torch


train_iter, valid_iter = get_data()
params = Word2VecParams()
tokenizer = get_tokenizer(params.TOKENIZER)
vocab = build_vocab(train_iter, tokenizer, params)
skipgrams = SkipGrams(vocab, params, tokenizer)
model = Model(vocab, params)
optimizer = torch.optim.Adam(params=model.parameters())

trainer = Trainer(
    model=model,
    params=params,
    optimizer=optimizer,
    vocab=vocab,
    train_iter=train_iter,
    valid_iter=valid_iter,
    skipgrams=skipgrams,
)

trainer.train()