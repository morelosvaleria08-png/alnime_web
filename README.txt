ALNIME Neural Demo (educational)
=================================

This demo shows a tiny Keras LSTM-based text model trained on a toy dataset.
It is for educational purposes only and does NOT provide psychological advice.
Run locally, do NOT expose to public without review.

How to run:
1. Create virtual env and activate it.
2. Install dependencies: pip install flask tensorflow
3. Run python train.py  (creates model files)
4. Run python app.py
5. Open http://127.0.0.1:5000 in your browser

Files:
- train.py : trains tiny model and saves alnime_model.h5, tokenizer.pkl, seq_len.pkl
- model.py : loads model and generates text continuation
- app.py   : Flask server that serves the chat and /chat endpoint
- templates/index.html : frontend chat UI
