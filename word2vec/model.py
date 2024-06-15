import torch.nn as nn
from vocab import Vocab
from params import Word2VecParams
import numpy as np
from scipy.spatial.distance import cosine


class Model(nn.Module):
    def __init__(self, vocab: Vocab, params: Word2VecParams):
        super().__init__()
        self.vocab = vocab
        # target word embedding
        self.t_embedding = nn.Embedding(
            self.vocab.__len__() + 1,
            params.EMBED_DIM,
            max_norm=params.EMBED_MAX_NORM,
        )  # V * N
        # context word embedding
        self.c_embedding = nn.Embedding(
            self.vocab.__len__() + 1,
            params.EMBED_DIM,
            max_norm=params.EMBED_MAX_NORM,
        )  #  V * N

    def forward(self, inputs, context):
        target_embeddings = self.t_embedding(inputs)
        n_examples = target_embeddings.shape[0]
        n_dimensions = target_embeddings.shape[1]
        target_embeddings = target_embeddings.view(
            n_examples, 1, n_dimensions
        )  # batch * 1 * N

        # negative sampling
        context_embeddings = self.c_embedding(
            context
        )  # batch * context size * N
        context_embeddings = context_embeddings.permute(
            0, 2, 1
        )  # batch * N * context size

        dots = target_embeddings.bmm(
            context_embeddings
        )  # batch * 1 * context size
        dots = dots.view(dots.shape[0], dots.shape[2])  # batch * context size
        return dots

    def normalize_embeddings(self):
        embeddings = list(self.t_embedding.parameters())[0]
        embeddings = embeddings.cpu().detach().numpy()
        norms = (embeddings**2).sum(axis=1) ** (1 / 2)
        norms = norms.reshape(norms.shape[0], 1)
        return embeddings / norms

    def get_similar_words(self, word, n):
        word_id = self.vocab.get_index(word)
        if word_id == 0:
            print("Out of vocabulary word")
            return

        embedding_norms = self.normalize_embeddings()
        word_vec = embedding_norms[word_id]
        word_vec = np.reshape(word_vec, (word_vec.shape[0], 1))
        dists = np.matmul(embedding_norms, word_vec).flatten()
        topN_ids = np.argsort(-dists)[1 : n + 1]

        topN_dict = {}
        for sim_word_id in topN_ids:
            sim_word = self.vocab.lookup_token(sim_word_id)
            topN_dict[sim_word] = dists[sim_word_id]
        return topN_dict

    def get_similarity(self, word1, word2):
        idx1 = self.vocab.get_index(word1)
        idx2 = self.vocab.get_index(word2)

        embedding_norms = self.normalize_embeddings()

        embedding1 = embedding_norms[idx1]
        embedding2 = embedding_norms[idx2]

        return cosine(embedding1, embedding2)
