import os
import torchdata.datapipes as dp
from functools import partial

_EXTRACTED_FILES = {
    "train": os.path.join("wikitext-103", "wiki.train.tokens"),
    "test": os.path.join("wikitext-103", "wiki.test.tokens"),
    "valid": os.path.join("wikitext-103", "wiki.valid.tokens"),
}
_DATASETS_DIR = "/home/feiyangc/datasets"

def load_wikitext103(split):
    file_path = os.path.join(_DATASETS_DIR, _EXTRACTED_FILES[split])
    print(file_path)
    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode="rb")
    # data_pipe = dp.iter.FileOpener(data_pipe, encoding="utf-8")
    return (
        data_pipe.readlines(strip_newline=False, return_path=False)
        .shuffle()
        .set_shuffle(False)
        .sharding_filter()
    )
