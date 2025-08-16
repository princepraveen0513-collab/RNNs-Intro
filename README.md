# RNN/LSTM — 3‑Part Project (Step‑by‑Step)

This README combines **Assignment06_RNN_Part1/Part2/Part3** into a single guide for building and improving a recurrent model for text classification.

- Dataset link: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set.

---

## 📁 Repository Layout
```
.
├── Assignment06_RNN_Part1.ipynb    # Step 1: Baseline model & pipeline
├── Assignment06_RNN_Part2.ipynb    # Step 2: Embeddings & sequence preprocessing
├── Assignment06_RNN_Part3.ipynb    # Step 3: Regularization, tuning & evaluation
└── README.md                        # This combined guide
```

---

## 🧰 Environment & Setup
**Dependencies:** `numpy`, `pandas`, `scikit-learn`, **one of**: (`torch`, `torchtext`) or (`tensorflow`, `keras`) plus `matplotlib` and `nltk`/`spacy` if used.

Install (PyTorch stack example):
```bash
pip install -U numpy pandas scikit-learn matplotlib nltk spacy torch torchvision torchtext
```

Install (TensorFlow/Keras stack example):
```bash
pip install -U numpy pandas scikit-learn matplotlib nltk spacy tensorflow
```

**Framework(s) detected:** PyTorch  
**Models referenced/used:** nn.LSTM  
**NLP prep detected:** Tokenizer, pad_sequences, Embedding  
**Device:** CUDA (if available) (auto‑detected in notebooks)

---

## ✅ Step 1 — Baseline RNN (Part 1)
**Notebook:** `Assignment06_RNN_Part1.ipynb`

- Load text data; clean/tokenize; build vocabulary.  
- Sequence preparation: **Tokenizer** → integer sequences → **pad_sequences** / truncation to `max_len`.  
- Build baseline model (**RNN/LSTM/GRU**) with optional **Embedding** layer.  
- Train for a few epochs; print train/val metrics.

Run:
```bash
jupyter notebook "Assignment06_RNN_Part1.ipynb"
```

---

## ⚙️ Step 2 — Embeddings & Pre‑Processing (Part 2)
**Notebook:** `Assignment06_RNN_Part2.ipynb`

- Improve preprocessing (lowercasing, punctuation removal, stopwords if used).  
- Try **pretrained embeddings** (e.g., GloVe/word2vec) or adjust **embedding_dim** and **max_len**.  
- Re‑train and compare metrics; save the best settings.

Run:
```bash
jupyter notebook "Assignment06_RNN_Part2.ipynb"
```

---

## 🚀 Step 3 — Regularization, Tuning & Evaluation (Part 3)
**Notebook:** `Assignment06_RNN_Part3.ipynb`

- Add **dropout**, **bidirectional** layers, and tune **hidden_size / num_layers**.  
- Consider **learning‑rate schedules** and different **optimizers** (Adam/SGD).  
- Evaluate on the test set; log **confusion matrix** and **classification report**.

Run:
```bash
jupyter notebook "Assignment06_RNN_Part3.ipynb"
```

---

## 🔧 Hyperparameters
- **hidden_size**: 64
- **num_layers**: 2
- **dropout**: 0.5
- **batch_size**: 64
- **lr**: 0.005

---

## 🧭 Next Steps
- Add **attention** or **self‑attention** layers for better context capture.  
- Try **subword tokenization** (WordPiece/BPE) or move to **transformers** (e.g., DistilBERT).  
- Perform **threshold tuning** for imbalanced datasets; monitor **PR‑AUC**.  
- Export the best model and tokenizer; provide an **inference** snippet in the README.
