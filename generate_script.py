import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from fetched_script_data import corpus
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

OOV_TOKEN = '<OOV>'
EPOCHS = 20

tokenizer = Tokenizer(oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

print('a2')
corpus = None  #GC

max_sequence_len = max([len(x) for x in input_sequences])
print('max seq len')
print(max_sequence_len)
padded = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
input_sequences = np.array(padded)
padded = None  #GC

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
labels = None  # GC
input_sequences = None  #GC

print('a3')

model = k.Sequential()
model.add(k.layers.Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(k.layers.Bidirectional(k.layers.LSTM(15)))
model.add(k.layers.Dense(total_words, activation='softmax'))
adam = k.optimizers.Adam(learning_rate=0.01)

print('a4')

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print('a5')

model.fit(xs, ys, epochs=EPOCHS, verbose=1)
adam = None  #GC
xs = None  #GC
ys = None  #GC

seed_text = "RACHEL: I love you Ross"
next_words = 300

print('Generate from seed text:')
print(seed_text)

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # predicted = model.predict_classes(token_list, verbose=0)
    predict_x = model.predict(token_list, verbose=0)
    predicted = np.argmax(predict_x, axis=1)
    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break

    seed_text += " " + output_word

print('Generated text:')
print(seed_text)
print('END')
