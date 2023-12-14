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

        # Add a special token for space
        self.alphabet.append(' ')

        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}

        if self.max_length is None:
            self.max_length = max([len(seq) for seq in data])

    def transform(self, data):
        data = [seq[:self.max_length] for seq in data]
        data = [self.padder(seq) for seq in data]

        # Update to handle spaces
        data = [[self.char_to_index[char] if char in self.char_to_index else self.char_to_index[' '] for char in seq]
                for seq in data]

        data = [to_categorical(seq, num_classes=len(self.alphabet)) for seq in data]

        return np.array(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        data = np.argmax(data, axis=-1)
        data = [[self.index_to_char[index] for index in seq] for seq in data]
        return data


if __name__ == '__main__':
    sequences = ['abc', 'defg', 'hij', 'klmnop']

    encoder = OneHotEncoder(padder=lambda x: x + ' ' * (encoder.max_length - len(x)), max_length=None)

    encoded_data = encoder.fit_transform(sequences)

    print("Encoded Data:")
    print(encoded_data)

    decoded_data = encoder.inverse_transform(encoded_data)

    print("\nDecoded Data:")
    print(decoded_data)
