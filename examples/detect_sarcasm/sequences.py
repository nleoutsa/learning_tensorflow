import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from examples.detect_sarcasm.model import compile_model

VOCAB_SIZE = 100000
TRAINING_SIZE = 22000
OOV_TOKEN = '<OOV>'


def tokenize(**kwargs):
    training_sentences = np.array(kwargs.get('sentences')[0:TRAINING_SIZE])
    testing_sentences = np.array(kwargs.get('sentences')[TRAINING_SIZE:])
    training_labels = np.array(kwargs.get('labels')[0:TRAINING_SIZE])
    testing_labels = np.array(kwargs.get('labels')[TRAINING_SIZE:])

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(kwargs.get('sentences'))

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, padding='post')

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, padding='post')

    model = compile_model(training_padded, training_labels, testing_padded, testing_labels, VOCAB_SIZE)

    new_headlines = [
        "The opening ceremony of the 2012 Olympics air tonight at 8PM",
        "Mass evacuations as wildfires spread in southern California",
        "Google announces acquisition of another Machine Learning company",
        "This just in -High School students don't like school",
        "Brave woman named Karen leaves store without asking to speak to the manager",
        "Senators see reason, vote to lower their own salaries"
    ]

    new_sequences = tokenizer.texts_to_sequences(new_headlines)
    padded = pad_sequences(new_sequences, padding='post')

    prediction = model.predict(padded)

    print("Headlines:")
    print('\n- '.join(new_headlines))

    print("\n\nPredictions:")
    print(prediction)