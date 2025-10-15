from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

model = load_model("alnime_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl","rb"))
max_sequence_len = pickle.load(open("seq_len.pkl","rb"))

def generate_text(seed_text, next_words=6):
    result = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        predicted = np.argmax(preds)
        # map index to word
        word = None
        for w, i in tokenizer.word_index.items():
            if i == predicted:
                word = w
                break
        if not word:
            break
        result += " " + word
    # simple cleanup
    return result

# small wrapper for integration
def alnime_reply(user_text):
    # use last 3 words as seed for generation (cleaned)
    seed = " ".join(user_text.strip().split()[-3:]) if user_text.strip() else "hola"
    gen = generate_text(seed, next_words=6)
    # If generation is basically echo or too short, return a safe canned fallback
    if gen.strip().lower() == user_text.strip().lower() or len(gen.split()) < 2:
        fallbacks = [
            "Interesante, cuéntame más.",
            "Gracias por compartir eso.",
            "¿Quieres hablar más sobre eso?"
        ]
        return np.random.choice(fallbacks)
    return gen
