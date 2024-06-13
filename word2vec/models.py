from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass(repr=True)
class Word2VecParams:
    # skipgram parameters
    MIN_FREQ = 50
    SKIPGRAM_N_WORDS = 8
    T = 85  #  subsampling words with frequency in the 85th percentile
    NEG_SAMPLES = 50  # num of negative samples to use for each training example
    NS_ARRAY_LEN = 5_000_000
    SPECIALS = ""  # placeholder for words that do not meet the minimum frequency requirement
    TOKENIZER = "basic_english"  # tokenize by splitting on spaces

    # training parameters
    BATCH_SIZE = 100
    EMBED_DIM = 300
    EMBED_MAX_NORM = None
    N_EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CRITERION = nn.BCEWithLogitsLoss()  # loss function