from vocab import Vocab
from params import Word2VecParams
import numpy as np
import random
import torch
import tqdm


class SkipGrams:
    def __init__(self, vocab: Vocab, params: Word2VecParams, tokenizer):
        self.vocab = vocab
        self.params = params
        self.t = self._t()
        self.tokenizer = tokenizer
        self.discard_probs = self._create_discard_dict()

    def _t(self):
        freq_list = []
        for _, (_, freq) in list(self.vocab.stoi.items())[1:]:
            freq_list.append(freq / self.vocab.total_tokens)
        return np.percentile(freq_list, self.params.T)

    def _create_discard_dict(self):
        discard_dict = {}
        for _, (word, freq) in self.vocab.stoi.items():
            discard_prob = 1 - np.sqrt(
                self.t / (freq / self.vocab.total_tokens + self.t)
            )
            discard_dict[word] = discard_prob

        return discard_dict

    def collate_skipgram(self, batch):
        batch_input, batch_output = [], []

        for text in batch:
            text_tokens = self.vocab.get_index(self.tokenizer(text))

            # sample too short for the context window
            if len(text_tokens) < self.params.SKIPGRAM_N_WORDS * 2 + 1:
                continue

            for idx in range(
                len(text_tokens) - self.params.SKIPGRAM_N_WORDS * 2 - 1
            ):
                token_id_sequence = text_tokens[
                    idx : idx + self.params.SKIPGRAM_N_WORDS * 2 + 1
                ]

                input = token_id_sequence.pop(self.params.SKIPGRAM_N_WORDS)
                outputs = token_id_sequence

                discard_prob = random.random()
                if input == 0 or self.discard_probs[input] >= discard_prob:
                    continue

                for output in outputs:
                    discard_prob = random.random()
                    if (
                        output == 0
                        or self.discard_probs[output] >= discard_prob
                    ):
                        continue

                    batch_input.append(input)
                    batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output


class NegativeSampler:
    def __init__(self, vocab: Vocab, ns_exponent: float, ns_array_len: int):
        self.vocab = vocab
        self.ns_exponent = ns_exponent
        self.ns_array_len = ns_array_len
        self.ns_array = self._create_negative_sampling()

    def __len__(self):
        return len(self.ns_array)

    def _create_negative_sampling(self):
        frequency_dict = {
            word_idx: freq**self.ns_exponent
            for _, (word_idx, freq) in list(self.vocab.stoi.items())[1:]
        }

        frequency_dict_scaled = {
            word: max(
                1, int((freq / self.vocab.total_tokens) * self.ns_array_len)
            )
            for word, freq in frequency_dict.items()
        }

        ns_array = []
        for word, freq in tqdm.tqdm(
            frequency_dict_scaled.items(), "Creating negative sampling array"
        ):
            ns_array = ns_array + [word] * freq
        return ns_array

    def sample(self, n_batches: int = 1, n_samples: int = 1):
        samples = []
        for _ in range(n_batches):
            samples.append(random.sample(self.ns_array, n_samples))
        samples = torch.as_tensor(np.array(samples))
        return samples
