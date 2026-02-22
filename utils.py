import random
import json
import gzip
import requests
import torch

GENRE_URLS = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


def load_reviews(url, limit=1500):
    reviews = []
    response = requests.get(url, stream=True)

    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i >= limit:
                break
            data = json.loads(line)
            reviews.append(data["review_text"])

    return reviews


def prepare_datasets():
    genre_reviews = {g: load_reviews(url) for g, url in GENRE_URLS.items()}

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():
        reviews = random.sample(reviews, 1000)

        train = reviews[:800]
        test = reviews[800:]

        train_texts.extend(train)
        test_texts.extend(test)
        train_labels.extend([genre] * len(train))
        test_labels.extend([genre] * len(test))

    return train_texts, train_labels, test_texts, test_labels


class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)