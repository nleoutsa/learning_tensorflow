import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.sequence as sequence


def get_sequences():
    sentences = [
        'I love my dog',
        'I love my cat',
        'You love my dog!',
        'Do you love my dog?',
        'Do you think my dog is amazing?'
    ];

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    new_sentences = [
        'I love my dog',
        'I love my MTV',
        'I want my MTV',
        'My MTV wants me to love my dog',
        'I love my dog who loves MTV'
    ];

    sequences = tokenizer.texts_to_sequences(new_sentences)
    padded = sequence.pad_sequences(sequences, padding='pre', maxlen=5, truncating='pre')

    print(word_index)
    print(sequences)
    print(padded)
