from model import Model
from params import Word2VecParams
from vocab import Vocab
from preprocessing import SkipGrams, NegativeSampler
from torch.utils.data import DataLoader
from time import monotonic


class Trainer:
    def __init__(
        self,
        model: Model,
        params: Word2VecParams,
        optimizer,
        vocab: Vocab,
        train_iter,
        valid_iter,
        skipgrams: SkipGrams,
    ):
        self.model = model
        self.optimizer = optimizer
        self.vocab = vocab
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.skipgrams = skipgrams
        self.params = params

        self.epoch_train_mins = {}
        self.loss = {"train": [], "valid": []}

        self.model.to(self.params.DEVICE)
        self.params.CRITERION.to(self.params.DEVICE)

        self.negative_sampler = NegativeSampler(
            vocab, ns_exponent=0.75, ns_array_len=self.params.NS_ARRAY_LEN
        )
        self.testwords = ["love", "hurricane", "military", "army"]

    def train(self):
        for epoch in range(self.params.N_EPOCHS):
            self.train_dataloader = DataLoader(
                self.train_iter,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False,
                collate_fn=self.skipgrams.collate_skipgram,
            )

            self.valid_dataLoader = DataLoader(
                self.valid_iter,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False,
                collate_fn=self.skipgrams.collate_skipgram,
            )

            st_time = monotonic()
            self._train_epoch()
            self.epoch_train_mins[epoch] = round(
                (monotonic() - st_time) / 60, 1
            )
            self._validate_epoch()

            print(
                f"""Epoch: {epoch+1}/{self.params.N_EPOCHS}\n""",
                f"""    Train Loss: {self.loss['train'][-1]:.2}\n""",
                f"""    Valid Loss: {self.loss['valid'][-1]:.2}\n""",
                f"""    Training Time (mins): {self.epoch_train_mins.get(epoch)}"""
                """\n""",
            )
            self.test_testwords()

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            if len(batch_data[0]) == 0:
                continue

            inputs = batch_data[0].to(self.params.DEVICE)
            pos_labels = batch_data[1].to(self.params.DEVICE)
            neg_labels = self.negative_sampler.sample(
                pos_labels.shape[0], self.params.NEG_SAMPLES
            )
