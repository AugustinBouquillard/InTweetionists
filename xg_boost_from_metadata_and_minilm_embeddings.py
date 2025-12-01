# Full pipeline: JSONL -> features -> Jina-V3 embeddings -> PCA -> LightGBM -> submission
# Paste & run in Colab

# Optional installs (uncomment if needed in Colab)
# !pip install -q transformers sentence-transformers accelerate lightgbm==3.4.1 textblob nltk

import os
import json
import math
import random
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from pandas import json_normalize
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import nltk
from textblob import TextBlob
import re

# ------------------------------
# 0. Settings
# ------------------------------
MODEL_NAME = "jinaai/jina-embeddings-v3"
EMB_DTYPE = np.float16              # on-disk storage for raw embeddings
PCA_DIM = 256                       # compressed embedding dim
EMB_BATCH = 64                      # embedding batch size (reduce if OOM)
PCA_BATCH = 1024                    # rows per incremental PCA partial_fit
TFIDF_TWEET_MAXFEAT = 2000         # reduce for memory
TFIDF_BIO_MAXFEAT = 2000
MEMMAP_DIR = "/content/emb_memmaps"
os.makedirs(MEMMAP_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ------------------------------
# 1. Robust JSONL loader
# ------------------------------
def load_jsonl_skip_bad(path):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed line silently
                continue
    return json_normalize(data_list)

print("Loading data...")
train_df = load_jsonl_skip_bad("train.jsonl")
kaggle_df = load_jsonl_skip_bad("kaggle_test.jsonl")
print("Loaded rows:", len(train_df), "train /", len(kaggle_df), "kaggle")

# Ensure label exists
if "label" not in train_df.columns:
    raise RuntimeError("train.jsonl must contain 'label' column")

# create X/y
X_train_raw = train_df.copy()
y_train = X_train_raw["label"].astype(int)
X_kaggle_raw = kaggle_df.copy()

# ------------------------------
# 2. Feature engineering: advanced metadata (your function, slightly hardened)
# ------------------------------
def create_advanced_features(df_input):
    df = df_input.copy()
    # fallback series
    default_int_series = pd.Series(0, index=df.index)
    default_bool_series = pd.Series(False, index=df.index)

    # numeric columns with safe get
    df["user.followers_count"] = df.get("user.followers_count", default_int_series).fillna(0).astype(float)
    df["user.friends_count"] = df.get("user.friends_count", default_int_series).fillna(0).astype(float)
    df["user.listed_count"] = df.get("user.listed_count", default_int_series).fillna(0).astype(float)
    df["user.favourites_count"] = df.get("user.favourites_count", default_int_series).fillna(0).astype(float)
    df["user.statuses_count"] = df.get("user.statuses_count", default_int_series).fillna(0).astype(float)
    df["retweet_count"] = df.get("retweet_count", default_int_series).fillna(0).astype(float)
    df["favorite_count"] = df.get("favorite_count", default_int_series).fillna(0).astype(float)
    df["quote_count"] = df.get("quote_count", default_int_series).fillna(0).astype(float)
    df["reply_count"] = df.get("reply_count", default_int_series).fillna(0).astype(float)

    # dates
    df["user_created_at_dt"] = pd.to_datetime(df.get("user.created_at", None), errors="coerce")
    ref_date = pd.to_datetime("now", utc=True)
    df["account_age_days"] = (ref_date - df["user_created_at_dt"]).dt.days.fillna(0).astype(float)

    df["created_at_dt"] = pd.to_datetime(df.get("created_at", None), errors="coerce")
    df["tweet_hour"] = df["created_at_dt"].dt.hour.fillna(-1).astype(int)
    df["tweet_is_weekend"] = df["created_at_dt"].dt.dayofweek.isin([5,6]).fillna(False).astype(int)

    # booleans
    df["is_default_profile"] = df.get("user.default_profile", default_bool_series).fillna(False).astype(int)
    df["is_default_image"] = df.get("user.default_profile_image", default_bool_series).fillna(False).astype(int)
    df["is_verified"] = df.get("user.verified", default_bool_series).fillna(False).astype(int)
    df["is_protected"] = df.get("user.protected", default_bool_series).fillna(False).astype(int)
    df["has_url"] = df.get("user.url", pd.Series(None, index=df.index)).notna().astype(int)

    # entities counting
    def count_entities(x):
        if isinstance(x, list):
            return len(x)
        return 0
    df["num_urls"] = df.get("entities.urls", default_int_series).apply(lambda x: count_entities(x)).astype(float)
    df["num_hashtags"] = df.get("entities.hashtags", default_int_series).apply(lambda x: count_entities(x)).astype(float)
    df["num_mentions"] = df.get("entities.user_mentions", default_int_series).apply(lambda x: count_entities(x)).astype(float)
    df["has_media"] = df.get("extended_entities.media", default_bool_series).notna().astype(int)

    # ratios
    followers = df["user.followers_count"].replace(0, 0.0)
    friends = df["user.friends_count"].replace(0, 0.0)
    listed = df["user.listed_count"].replace(0, 0.0)
    statuses = df["user.statuses_count"].replace(0, 0.0)

    df["ratio_followers_friends"] = followers / (friends + 1.0)
    df["ratio_listed_followers"] = listed / (followers + 1.0)
    df["reciprocity_score"] = (friends - followers) / (friends + followers + 1.0)
    df["tweets_per_day"] = statuses / (df["account_age_days"] + 1.0)
    df["ratio_mention_status"] = df["num_mentions"] / (statuses + 1.0)
    total_engagement = df["retweet_count"] + df["favorite_count"] + df["quote_count"] + df["reply_count"]
    df["total_tweet_engagement"] = total_engagement / (followers + 1.0)

    # text lengths and final_text
    df["final_text"] = df.get("extended_tweet.full_text", df.get("text", "")).fillna("")
    df["final_text"] = df["final_text"].where(df["final_text"] != "", df.get("text", "")).fillna("")
    df["text_length"] = df["final_text"].astype(str).apply(len).astype(float)
    df["bio_length"] = df.get("user.description", "").fillna("").astype(str).apply(len).astype(float)

    features_to_keep = [
        'user.followers_count','user.friends_count','user.listed_count','user.favourites_count','user.statuses_count',
        'retweet_count','favorite_count','quote_count','reply_count',
        'ratio_followers_friends','ratio_listed_followers','tweets_per_day','account_age_days',
        'reciprocity_score','ratio_mention_status','total_tweet_engagement',
        'is_verified','is_default_profile','is_default_image','is_protected','has_url',
        'tweet_hour','tweet_is_weekend','text_length','bio_length',
        'num_urls','num_hashtags','num_mentions','has_media'
    ]

    final_cols = [c for c in features_to_keep if c in df.columns]
    return df[final_cols].fillna(0)

print("Building advanced features...")
X_train_meta = create_advanced_features(X_train_raw)
X_kaggle_meta = create_advanced_features(X_kaggle_raw)
print("Meta shapes:", X_train_meta.shape, X_kaggle_meta.shape)

# ------------------------------
# 3. NLP meta-features: TF-IDF + meta logistic models + TextBlob sentiment
# ------------------------------
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
french_stopwords = stopwords.words("french")

def get_final_text_series(df):
    text = df.get("extended_tweet.full_text", df.get("text", pd.Series([""] * len(df))))
    return text.fillna("").astype(str)

print("Preparing TF-IDF meta-features...")
train_texts_for_tfidf = get_final_text_series(X_train_raw)
kaggle_texts_for_tfidf = get_final_text_series(X_kaggle_raw)

# tweet TF-IDF (char_wb as you had)
tfidf_tweet = TfidfVectorizer(max_features=TFIDF_TWEET_MAXFEAT, stop_words=french_stopwords, ngram_range=(2,5), analyzer='char_wb', lowercase=False)
X_tfidf_train_tweet = tfidf_tweet.fit_transform(train_texts_for_tfidf)
X_tfidf_kaggle_tweet = tfidf_tweet.transform(kaggle_texts_for_tfidf)

# train meta logistic on tweet tfidf
logreg_tweet = LogisticRegression(solver="sag", max_iter=1000)
logreg_tweet.fit(X_tfidf_train_tweet, y_train)
train_tweet_proba = logreg_tweet.predict_proba(X_tfidf_train_tweet)[:,1]
kaggle_tweet_proba = logreg_tweet.predict_proba(X_tfidf_kaggle_tweet)[:,1]

# bio TF-IDF
train_bio = X_train_raw.get("user.description", pd.Series([""] * len(X_train_raw))).fillna("").astype(str)
kaggle_bio = X_kaggle_raw.get("user.description", pd.Series([""] * len(X_kaggle_raw))).fillna("").astype(str)

tfidf_bio = TfidfVectorizer(max_features=TFIDF_BIO_MAXFEAT, stop_words=french_stopwords, ngram_range=(2,5), analyzer='char_wb', lowercase=False)
X_tfidf_train_bio = tfidf_bio.fit_transform(train_bio)
X_tfidf_kaggle_bio = tfidf_bio.transform(kaggle_bio)

logreg_bio = LogisticRegression(solver="liblinear", max_iter=1000)
logreg_bio.fit(X_tfidf_train_bio, y_train)
train_bio_proba = logreg_bio.predict_proba(X_tfidf_train_bio)[:,1]
kaggle_bio_proba = logreg_bio.predict_proba(X_tfidf_kaggle_bio)[:,1]

# sentiment using TextBlob (note: English-oriented, but works reasonably)
def get_sentiment_series(series):
    out_polarity = []
    out_subjectivity = []
    for txt in tqdm(series, desc="Sentiment", leave=False):
        try:
            tb = TextBlob(str(txt))
            out_polarity.append(tb.sentiment.polarity)
            out_subjectivity.append(tb.sentiment.subjectivity)
        except:
            out_polarity.append(0.0)
            out_subjectivity.append(0.0)
    return np.array(out_polarity), np.array(out_subjectivity)

train_tweet_polarity, train_tweet_subjectivity = get_sentiment_series(train_texts_for_tfidf)
kaggle_tweet_polarity, kaggle_tweet_subjectivity = get_sentiment_series(kaggle_texts_for_tfidf)

# Collect NLP meta-features into DataFrames
X_train_nlp_meta = pd.DataFrame({
    "meta_bio_proba": train_bio_proba,
    "meta_tweet_proba": train_tweet_proba,
    "tweet_polarity": train_tweet_polarity,
    "tweet_subjectivity": train_tweet_subjectivity
})

X_kaggle_nlp_meta = pd.DataFrame({
    "meta_bio_proba": kaggle_bio_proba,
    "meta_tweet_proba": kaggle_tweet_proba,
    "tweet_polarity": kaggle_tweet_polarity,
    "tweet_subjectivity": kaggle_tweet_subjectivity
})

print("NLP meta features shapes:", X_train_nlp_meta.shape, X_kaggle_nlp_meta.shape)

# Merge metadata + nlp meta
X_train_meta = pd.concat([X_train_meta.reset_index(drop=True), X_train_nlp_meta.reset_index(drop=True)], axis=1)
X_kaggle_meta = pd.concat([X_kaggle_meta.reset_index(drop=True), X_kaggle_nlp_meta.reset_index(drop=True)], axis=1)

# ------------------------------
# 4. Prepare text lists for embeddings (final_text)
# ------------------------------
def get_final_text_list(df):
    if "extended_tweet.full_text" in df.columns:
        s = df["extended_tweet.full_text"].fillna("").astype(str)
    else:
        s = df.get("text", pd.Series([""] * len(df))).fillna("").astype(str)
    return s.tolist()

train_texts = get_final_text_list(X_train_raw)
kaggle_texts = get_final_text_list(X_kaggle_raw)
print("Texts ready:", len(train_texts), len(kaggle_texts))

# ------------------------------
# 5. Load Jina model (encoder) for embeddings
# ------------------------------
print("Loading embedding model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
torch_dtype = torch.float16 if DEVICE == "cuda" else None
embedder = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype)
embedder = embedder.to(DEVICE)
embedder.eval()
if DEVICE == "cuda":
    try:
        embedder.half()
    except Exception:
        pass

EMB_SIZE = embedder.config.hidden_size
print("Embedder hidden size:", EMB_SIZE)

# ------------------------------
# 6. Stream embeddings to memmap (train + kaggle)
# ------------------------------
def make_memmap(path, shape, dtype=np.float16):
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)

train_emb_path = os.path.join(MEMMAP_DIR, "train_emb.dat")
kaggle_emb_path = os.path.join(MEMMAP_DIR, "kaggle_emb.dat")

train_mm = make_memmap(train_emb_path, (len(train_texts), EMB_SIZE), dtype=EMB_DTYPE)
kaggle_mm = make_memmap(kaggle_emb_path, (len(kaggle_texts), EMB_SIZE), dtype=EMB_DTYPE)

def mean_pooling(model_output, attention_mask):
    token_emb = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).to(token_emb.dtype)
    summed = (token_emb * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / denom)

def stream_embed(texts, memmap_arr, batch_size=EMB_BATCH):
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {k:v.to(DEVICE) for k,v in encoded.items()}
        with torch.no_grad():
            out = embedder(**encoded)
            pooled = mean_pooling(out, encoded["attention_mask"]).to("cpu").numpy()
        # normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        pooled = pooled / norms
        memmap_arr[start:end, :] = pooled.astype(EMB_DTYPE)
        torch.cuda.empty_cache()
    memmap_arr.flush()

print("Streaming train embeddings...")
stream_embed(train_texts, train_mm, batch_size=EMB_BATCH)
print("Streaming kaggle embeddings...")
stream_embed(kaggle_texts, kaggle_mm, batch_size=EMB_BATCH)

# reopen as read-only memmaps
train_embs = np.memmap(train_emb_path, dtype=EMB_DTYPE, mode="r", shape=(len(train_texts), EMB_SIZE))
kaggle_embs = np.memmap(kaggle_emb_path, dtype=EMB_DTYPE, mode="r", shape=(len(kaggle_texts), EMB_SIZE))

# ------------------------------
# 7. Incremental PCA to compress embeddings -> PCA_DIM
# ------------------------------
print("Running IncrementalPCA to", PCA_DIM, "dims")
ipca = IncrementalPCA(n_components=PCA_DIM, batch_size=PCA_BATCH)

for start in tqdm(range(0, train_embs.shape[0], PCA_BATCH), desc="IPCA fit"):
    end = min(start + PCA_BATCH, train_embs.shape[0])
    chunk = np.asarray(train_embs[start:end]).astype(np.float32)
    ipca.partial_fit(chunk)

def ipca_transform_to_memmap(src_memmap, out_path, n_components=PCA_DIM, batch_size=PCA_BATCH):
    out_mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(src_memmap.shape[0], n_components))
    for start in tqdm(range(0, src_memmap.shape[0], batch_size), desc=f"IPCA transform {os.path.basename(out_path)}"):
        end = min(start + batch_size, src_memmap.shape[0])
        chunk = np.asarray(src_memmap[start:end]).astype(np.float32)
        out_mm[start:end] = ipca.transform(chunk)
    out_mm.flush()
    return out_mm

train_pca_path = os.path.join(MEMMAP_DIR, "train_emb_pca.npy")
kaggle_pca_path = os.path.join(MEMMAP_DIR, "kaggle_emb_pca.npy")
train_pca_mm = ipca_transform_to_memmap(train_embs, train_pca_path, n_components=PCA_DIM)
kaggle_pca_mm = ipca_transform_to_memmap(kaggle_embs, kaggle_pca_path, n_components=PCA_DIM)

train_pca = np.memmap(train_pca_path, dtype=np.float32, mode="r", shape=(len(train_texts), PCA_DIM))
kaggle_pca = np.memmap(kaggle_pca_path, dtype=np.float32, mode="r", shape=(len(kaggle_texts), PCA_DIM))

print("PCA shapes:", train_pca.shape, kaggle_pca.shape)

# ------------------------------
# 8. Combine sparse TF-IDF (tweet + bio) + meta features + PCA embeddings
# We'll convert PCA embeddings to dense csr to hstack with sparse TFIDF.
# ------------------------------
print("Combining features...")

# Build sparse TFIDF for tweets (we already have tweet tfidf matrices)
# Combine tweet tfidf and bio tfidf? We used both as meta logit models, but we can also include raw TFIDF.
# For memory reasons, we'll include the tweet TF-IDF sparse matrix only (X_tfidf_train_tweet).
# If you want, change to include bio tfidf as well.

# Create CSR for PCA embeddings
train_pca_csr = csr_matrix(train_pca)
kaggle_pca_csr = csr_matrix(kaggle_pca)

# Build final sparse matrices: [tweet_tfidf | pca_emb | sentiment/meta small dense]
# First build sentiment/meta sparse matrix
meta_train_dense = X_train_meta.values.astype(np.float32)
meta_kaggle_dense = X_kaggle_meta.values.astype(np.float32)

# Convert dense meta to csr
meta_train_csr = csr_matrix(meta_train_dense)
meta_kaggle_csr = csr_matrix(meta_kaggle_dense)

# Also include the meta logistic probs & polarity/subjectivity we computed earlier are already in meta.

X_train_sparse = hstack([X_tfidf_train_tweet, train_pca_csr, meta_train_csr], format="csr")
X_kaggle_sparse = hstack([X_tfidf_kaggle_tweet, kaggle_pca_csr, meta_kaggle_csr], format="csr")

print("Final sparse shapes:", X_train_sparse.shape, X_kaggle_sparse.shape)

# ------------------------------
# 9. Train/validation split (use small val for early stopping)
# ------------------------------
print("Splitting train/val...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train_sparse, y_train.values, test_size=0.12, random_state=42, stratify=y_train)

# ------------------------------
# 10. LightGBM dataset & training with early stopping
# ------------------------------
print("Training LightGBM (with early stopping)...")
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": 4
}

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

bst = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dval],
    early_stopping_rounds=50,
    verbose_eval=50
)

# ------------------------------
# 11. Predict on Kaggle
# ------------------------------
print("Predicting Kaggle set...")
preds_prob = bst.predict(X_kaggle_sparse, num_iteration=bst.best_iteration)
preds_label = (preds_prob >= 0.5).astype(int)

submission = pd.DataFrame({
    "ID": X_kaggle_raw.get("challenge_id", X_kaggle_raw.index).reset_index(drop=True),
    "Prediction": preds_label
})

out_path = "submission_jina_v3_lightgbm.csv"
submission.to_csv(out_path, index=False)
print("Saved submission:", out_path, "shape:", submission.shape)

# optional download (Colab)
try:
    from google.colab import files
    files.download(out_path)
except Exception:
    pass

print("Pipeline complete.")
