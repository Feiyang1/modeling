import numpy as np
from typing import Union, List, Optional
import re
from params import Word2VecParams
from collections import Counter, OrderedDict


class Vocab:
    def __init__(self, vocabs, specials):
        self.stoi = {v[0]: (i, v[1]) for i, v in enumerate(vocabs)}
        self.itos = {i: (v[0], v[1]) for i, v in enumerate(vocabs)}
        self._specials = specials[0]
        self.total_tokens = np.nansum(
            [f for _, (_, f) in self.stoi.items()], dtype=int
        )

    def __len__(self):
        return len(self.stoi) - 1

    def get_index(self, word: Union[str, List]):
        if isinstance(word, str):
            if word in self.stoi:
                return self.stoi.get(word)[0]
            else:
                return self.stoi.get(self._specials)[0]
        elif isinstance(word, list):
            res = []
            for w in word:
                if w in self.stoi:
                    res.append(self.stoi.get(w)[0])
                else:
                    res.append(self.stoi.get(self._specials)[0])
            return res
        else:
            raise ValueError(
                f"Word {word} is not a string or a list of strings."
            )

    def get_freq(self, word: Union[str, List]):
        if isinstance(word, str):
            if word in self.stoi:
                return self.stoi.get(word)[1]
            else:
                return self.stoi.get(self._specials)[1]
        elif isinstance(word, list):
            res = []
            for w in word:
                if w in self.stoi:
                    res.append(self.stoi.get(w)[1])
                else:
                    res.append(self.stoi.get(self._specials)[1])
            return res
        else:
            raise ValueError(
                f"Word {word} is not a string or a list of strings."
            )

    def lookup_token(self, token: Union[int, List]):
        if isinstance(token, (int, np.int64)):
            if token in self.itos:
                return self.itos.get(token)[0]
            else:
                raise ValueError(f"Token {token} not in vocabulary")
        elif isinstance(token, list):
            res = []
            for t in token:
                if t in self.itos:
                    res.append(self.itos.get(token)[0])
                else:
                    raise ValueError(f"Token {t} is not a valid index.")
            return res


def yield_tokens(iterator, tokenizer):
    r = re.compile("[a-z1-9]")
    for text in iterator:
        res = tokenizer(text)
        res = list(filter(r.match, res))
        yield res


def build_vocab(
    iterator,
    tokenizer,
    params: Word2VecParams,
    max_tokens: Optional[int] = None,
):
    counter = Counter()
    for tokens in yield_tokens(iterator, tokenizer):
        counter.update(tokens)

    # sort by frequency in descending order, then lexicographically
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    tokens = []

    for token, freq in ordered_dict.items():
        if freq > params.MIN_FREQ:
            tokens.append((token, freq))

    specials = (params.SPECIALS, np.nan)
    tokens[0] = (
        specials  # TODO: why does it overwrite the first element instead of appending to the end
    )
    return Vocab(tokens, specials)
