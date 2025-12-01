# ================================================
# FULL TWEET CLASSIFIER PIPELINE
# CamemBERT Embeddings + TFIDF + Sentiment + LightGBM
# ================================================

!pip install transformers sentencepiece lightgbm emoji datasets --quiet

import pandas as pd
import numpy as np
import torch
from transformers import CamembertModel, CamembertTokenizerFast, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from scipy.sparse import hstack, csr_matrix
import emoji
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Device:", device)

# ================================================
# 1. Load dataset (robust JSONL tolerant loader)
# ================================================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(pd.read_json(line, typ="series"))
            except:
                continue
    return pd.DataFrame(data)

train_df = load_jsonl("train.jsonl")
test_df = load_jsonl("test.jsonl")

# Extract nested tweet content
train_df["text"] = train_df["tweet"].apply(lambda x: x.get("text", "") if isinstance(x, dict) else "")
test_df["text"] = test_df["tweet"].apply(lambda x: x.get("text", "") if isinstance(x, dict) else "")

y = train_df["label"]
X_train_text = train_df["text"].fillna("")
X_test_text = test_df["text"].fillna("")

# ================================================
# 2. TF-IDF Features
# ================================================
tfidf = TfidfVectorizer(
    max_features=10_000,
    ngram_range=(1,2),
    min_df=3
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# ================================================
# 3. Sentiment Features (French capable)
# ================================================
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def get_sentiment_scores(texts):
    res = sentiment(texts)
    scores = []
    for r in res:
        if isinstance(r, dict):
            s = r.get("score", 0)
            label = r.get("label", "NEU")
            scores.append([ 
                s if label=="NEG" else 0,
                s if label=="POS" else 0,
                s if label=="NEU" else 0
            ])
        else:
            scores.append([0,0,0])
    return np.array(scores)

print("🔎 Sentiment analysis...")
train_sent = get_sentiment_scores(X_train_text.tolist())
test_sent = get_sentiment_scores(X_test_text.tolist())

# ================================================
# 4. CamemBERT Embeddings (pooled output)
# ================================================
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base").to(device)

# light embedding
def camembert_embed(batch):
    enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k:v.to(device) for k,v in enc.items()})
    # CLS embedding: pooled output
    return out.last_hidden_state[:,0,:].cpu().numpy()  # shape = (batch,768)

def embed_all(texts):
    BATCH = 32
    vectors = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        vec = camembert_embed(chunk)
        vectors.append(vec)
    return np.vstack(vectors)

print("🔄 Embedding with CamemBERT...")
X_train_cam = embed_all(X_train_text.tolist())
X_test_cam = embed_all(X_test_text.tolist())

# Standardize CamemBERT embeddings
scaler = StandardScaler()
X_train_cam = scaler.fit_transform(X_train_cam)
X_test_cam = scaler.transform(X_test_cam)

# Convert dense → sparse to concatenate with TF-IDF
X_train_cam_sp = csr_matrix(X_train_cam)
X_test_cam_sp = csr_matrix(X_test_cam)
train_sent_sp = csr_matrix(train_sent)
test_sent_sp = csr_matrix(test_sent)

# ================================================
# 5. Final Feature Matrix (TF-IDF + CamemBERT + Sentiment)
# ================================================
X_train_final = hstack([X_train_tfidf, X_train_cam_sp, train_sent_sp])
X_test_final  = hstack([X_test_tfidf,  X_test_cam_sp,  test_sent_sp])

print("🔢 Final feature shape:", X_train_final.shape)

# ================================================
# 6. Train LightGBM
# ================================================
clf = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.8
)

print("🚀 Training LightGBM...")
clf.fit(X_train_final, y)

# ================================================
# 7. Predict + Save Kaggle submission
# ================================================
preds = clf.predict(X_test_final)

submission = pd.DataFrame({
    "ID": test_df["challenge_id"],
    "Prediction": preds
})

submission.to_csv("submission.csv", index=False)

print("📄 Saved submission.csv")
