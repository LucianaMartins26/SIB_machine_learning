import numpy as np
from keras.utils import to_categorical


class OneHotEncoder:
    def __init__(self, padder, max_length):
        self.padder = padder
        self.max_length = max_length

        self.alphabet = None
        self.char_to_index = None
        self.index_to_char = None

    def fit(self, data):
        self.alphabet = set()
        for seq in data:
            self.alphabet.update(seq)

        self.alphabet = sorted(self.alphabet)

        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}

        if self.max_length is None:
            self.max_length = max([len(seq) for seq in data])

    def transform(self, data):

        data = [seq[:self.max_length] for seq in data]
        data = [self.padder(seq) for seq in data]

        data = [[self.char_to_index[char] for char in seq] for seq in data]

        data = [to_categorical(seq, num_classes=len(self.alphabet)) for seq in data]

        return np.array(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        data = np.argmax(data, axis=-1)
        data = [[self.index_to_char[index] for index in seq] for seq in data]
        return data

