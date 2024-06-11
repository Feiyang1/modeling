from torchtext.data import to_map_style_dataset
from data_loader import load_wikitext103


def get_data():
    train_iter = load_wikitext103(split="train")
    train_iter = to_map_style_dataset(train_iter)
    valid_iter = load_wikitext103(split="test")
    valid_iter = to_map_style_dataset(valid_iter)

    return train_iter, valid_iter


train_iter, valid_iter = get_data()

count = 0
for text in iter(train_iter):
    print(text)
    count += 1
    if count == 6:
        break
