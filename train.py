from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# WARNING: tiny dataset for demo purposes ONLY
data = [
    "hola",
    "hola como estas",
    "estoy bien gracias",
    "me siento triste hoy",
    "estoy muy feliz",
    "me siento ansioso por el examen",
    "estoy aburrida en clase",
    "me gusta platicar contigo",
    "cuentame algo",
    "necesito un consejo",
    "gracias me siento mejor"
]

# tokenizer and sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = np.zeros((len(labels), total_words))
for i, label in enumerate(labels):
    ys[i, label] = 1

model = Sequential()
model.add(Embedding(total_words, 8, input_length=max_sequence_len-1))
model.add(LSTM(64))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training model (demo, may take a minute)...")
model.fit(xs, ys, epochs=200, verbose=0)
model.save("alnime_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("seq_len.pkl", "wb") as f:
    pickle.dump(max_sequence_len, f)
print("Model and tokenizer saved: alnime_model.h5, tokenizer.pkl, seq_len.pkl")
