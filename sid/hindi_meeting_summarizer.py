#!/usr/bin/env python3
"""
Hindi Meeting Summarizer + Semantic Extractor
Copy-paste this file and run.

Requirements:
pip install torch sentencepiece sklearn nltk tqdm

Notes:
- Training seq2seq well requires (transcript, summary) pairs. If you don't have many labeled pairs,
  consider creating a small seed dataset or using an extractive pseudo-summary as targets to bootstrap.
- Training on GPU is strongly recommended.
"""

import os
import math
import random
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from tqdm import tqdm

# Download punkt for sentence tokenization (Hindi works reasonably)
nltk.download('punkt')

# ----------------------------
# Config / Hyperparameters
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_MODEL = "hindi_spm.model"
VOCAB_SIZE = 8000         # subword vocab size
MAX_SRC_LEN = 1024
MAX_TGT_LEN = 120
BATCH_SIZE = 8
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 1024
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
LR = 1e-4
EPOCHS = 10
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# ----------------------------
# Utilities: Tokenizer (SentencePiece)
# ----------------------------
def train_sentencepiece(input_txt_path: str, model_prefix: str = "hindi_spm", vocab_size: int = VOCAB_SIZE):
    """
    Train a SentencePiece tokenizer on a text file.
    input_txt_path: a single file with many Hindi sentences (one per line preferred).
    """
    spm.SentencePieceTrainer.train(
        input=input_txt_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,   # high coverage for Hindi
        model_type='unigram',        # or 'bpe'
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    print("Trained SentencePiece model:", model_prefix + ".model")

def load_tokenizer(model_path: str = TOKENIZER_MODEL):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

# ----------------------------
# Dataset
# ----------------------------
class MeetingSummaryDataset(Dataset):
    """
    dataset expects list of (transcript_text, summary_text) pairs for supervised training.
    If you don't have summaries, you can use extractive heuristics to create pseudo summaries.
    """
    def __init__(self, pairs: List[Tuple[str,str]], sp: spm.SentencePieceProcessor,
                 max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN):
        self.pairs = pairs
        self.sp = sp
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        # encode, return ids
        src_ids = self.sp.encode(src, out_type=int)[: self.max_src_len - 2]
        tgt_ids = self.sp.encode(tgt, out_type=int)[: self.max_tgt_len - 2]
        # add BOS/EOS (sentencepiece uses ids 2=bos, 3=eos due to train config above)
        src_ids = [self.sp.bos_id()] + src_ids + [self.sp.eos_id()]
        tgt_ids = [self.sp.bos_id()] + tgt_ids + [self.sp.eos_id()]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    pad_id = 0  # we assigned pad_id=0 in SentencePiece training
    src_batch = torch.full((len(srcs), max_src), pad_id, dtype=torch.long)
    tgt_batch = torch.full((len(tgts), max_tgt), pad_id, dtype=torch.long)
    for i, s in enumerate(srcs):
        src_batch[i, :len(s)] = s
    for i, t in enumerate(tgts):
        tgt_batch[i, :len(t)] = t
    return src_batch, tgt_batch

# ----------------------------
# Model: Transformer Seq2Seq using PyTorch's nn.Transformer
# ----------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size,
                 nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=FFN_HID_DIM, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=0)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=0)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.emb_size = emb_size

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        # src: (S, N) expected by nn.Transformer as seq_len x batch
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        outs = self.transformer.decoder(tgt_emb, memory,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)
        logits = self.generator(outs)
        return logits

    def encode(self, src, src_mask, src_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)  # (maxlen, 1, emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # token_embedding shape: seq_len, batch_size, emb_size
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# ----------------------------
# Mask helpers
# ----------------------------
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.size(0)
    tgt_seq_len = tgt.size(0)

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask, tgt_mask.to(DEVICE), src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)

# ----------------------------
# Training loop
# ----------------------------
def train_epoch(model, optimizer, dataloader, sp):
    model.train()
    losses = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for src_batch, tgt_batch in tqdm(dataloader, desc="Training batches"):
        src_batch = src_batch.transpose(0,1).to(DEVICE)    # seq_len, batch
        tgt_batch = tgt_batch.transpose(0,1).to(DEVICE)    # seq_len, batch
        tgt_input = tgt_batch[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_batch, tgt_input)
        memory_key_padding_mask = src_padding_mask

        optimizer.zero_grad()
        logits = model(src_batch, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # logits: seq_len x batch x vocab
        tgt_out = tgt_batch[1:, :].reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        loss = criterion(logits, tgt_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses += loss.item()
    return losses / len(dataloader)

# ----------------------------
# Inference / Greedy decode
# ----------------------------
@torch.no_grad()
def summarize(model, src_text: str, sp: spm.SentencePieceProcessor, max_len=MAX_TGT_LEN):
    model.eval()
    src_ids = sp.encode(src_text, out_type=int)[:MAX_SRC_LEN-2]
    src_ids = [sp.bos_id()] + src_ids + [sp.eos_id()]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(DEVICE)  # seq_len x 1
    src_mask = torch.zeros((src.size(0), src.size(0)), device=DEVICE).type(torch.bool)
    src_padding_mask = (src == 0).transpose(0,1)

    memory = model.encode(src, src_mask, src_padding_mask)
    ys = torch.tensor([sp.bos_id()], dtype=torch.long).unsqueeze(1).to(DEVICE)
    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0,1)  # batch x seq x emb
        prob = model.generator(out[:, -1, :])  # batch x vocab
        next_token = torch.argmax(prob, dim=1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=0)
        if next_token == sp.eos_id():
            break
    out_ids = ys.squeeze(1).tolist()
    # remove BOS and everything after EOS
    if out_ids and out_ids[0] == sp.bos_id():
        out_ids = out_ids[1:]
    if sp.eos_id() in out_ids:
        out_ids = out_ids[:out_ids.index(sp.eos_id())]
    summary = sp.decode(out_ids)
    return summary

# ----------------------------
# Semantic extraction: sentence embeddings and cluster-based topic extraction
# ----------------------------
def extract_semantic_insights(transcript_text: str, sp: spm.SentencePieceProcessor, model: Seq2SeqTransformer,
                              n_topics=3, top_n_sentences=3):
    """
    Returns:
      - representative_sentences: list of top_n_sentences that best represent clusters
      - keywords: top tf-idf keywords across transcript
    """
    # 1) Split transcript into sentences (nltk)
    sentences = nltk.tokenize.sent_tokenize(transcript_text)
    if len(sentences) == 0:
        return [], []
    # 2) Get sentence embeddings by encoding sentences and averaging token embeddings from encoder
    model.eval()
    sent_embeddings = []
    with torch.no_grad():
        for s in sentences:
            ids = sp.encode(s, out_type=int)[:MAX_SRC_LEN-2]
            ids = [sp.bos_id()] + ids + [sp.eos_id()]
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(1).to(DEVICE)
            src_padding_mask = (src == 0).transpose(0,1)
            src_mask = torch.zeros((src.size(0), src.size(0)), device=DEVICE).type(torch.bool)
            memory = model.encode(src, src_mask, src_padding_mask)  # seq_len x batch x emb
            # average over sequence dimension
            emb = memory.mean(dim=0).squeeze(0).cpu().numpy()  # emb_size
            sent_embeddings.append(emb)
    # 3) Cluster sentence embeddings to find representative sentences
    n_clusters = min(n_topics, len(sentences))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sent_embeddings)
    clusters = kmeans.labels_
    representative_sentences = []
    for c in range(n_clusters):
        idxs = [i for i, lab in enumerate(clusters) if lab == c]
        if not idxs:
            continue
        # pick the sentence closest to cluster center
        center = kmeans.cluster_centers_[c]
        best_idx = min(idxs, key=lambda i: np_l2(sent_embeddings[i], center))
        representative_sentences.append(sentences[best_idx])
    # 4) TF-IDF keywords
    vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1,2))
    try:
        tfidf = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        # average tfidf across sentences
        scores = tfidf.mean(axis=0).A1
        top_k_idx = scores.argsort()[::-1][:20]
        keywords = [feature_names[i] for i in top_k_idx]
    except Exception:
        keywords = []

    return representative_sentences[:top_n_sentences], keywords

def np_l2(a, b):
    import numpy as np
    return float(np.sum((np.array(a) - np.array(b))**2))

# ----------------------------
# Saving / Loading helpers
# ----------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Saved model to", path)

def load_model(path, src_vocab_size, tgt_vocab_size):
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                               NHEAD, src_vocab_size, tgt_vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

# ----------------------------
# Example usage & CLI-like helper
# ----------------------------
def build_dummy_pairs_from_transcript(transcript_file: str, sp: spm.SentencePieceProcessor, max_pairs=200):
    """
    If you don't have (transcript, summary) labeled pairs, this helper creates
    pseudo-supervised pairs by extracting top sentences as short summaries (extractive).
    Not ideal but can bootstrap training.
    """
    with open(transcript_file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = nltk.tokenize.sent_tokenize(text)
    # create pseudo summaries: for each chunk of ~8 sentences, take top 2 sentences by TF-IDF as summary
    pairs = []
    chunk_size = 8
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i+chunk_size])
        if not chunk.strip():
            continue
        # simple tfidf on the chunk
        vect = TfidfVectorizer(max_features=50, ngram_range=(1,2))
        try:
            tfidf = vect.fit_transform([chunk])
            # pick first 2 sentences as summary fallback
            summary = " ".join(sentences[i:i+2])
        except Exception:
            summary = " ".join(sentences[i:i+2])
        pairs.append((chunk, summary))
        if len(pairs) >= max_pairs:
            break
    return pairs

def main_train_flow(transcript_file: str, sp_model: str = TOKENIZER_MODEL, train_new_tokenizer=False):
    """
    High-level flow:
      - optionally train tokenizer on transcript
      - create pseudo pairs (if no gold summaries)
      - train transformer seq2seq
    """
    # 1) Train tokenizer if requested
    if train_new_tokenizer:
        print("Training SentencePiece tokenizer... (this will create hindi_spm.model)")
        train_sentencepiece(transcript_file, model_prefix="hindi_spm", vocab_size=VOCAB_SIZE)
        sp_path = "hindi_spm.model"
    else:
        sp_path = sp_model

    sp = load_tokenizer(sp_path)
    # 2) Build pseudo supervised pairs
    pairs = build_dummy_pairs_from_transcript(transcript_file, sp, max_pairs=200)
    dataset = MeetingSummaryDataset(pairs, sp)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    src_vocab_size = sp.get_piece_size()
    tgt_vocab_size = sp.get_piece_size()

    # 3) Build model
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                               NHEAD, src_vocab_size, tgt_vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 4) Train
    print("Starting training on", DEVICE)
    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(model, optimizer, dataloader, sp)
        print(f"Epoch {epoch}/{EPOCHS} average loss: {loss:.4f}")
        save_model(model, f"transformer_epoch{epoch}.pth")
    print("Training complete. Last model saved as transformer_epoch{EPOCHS}.pth")
    return model, sp

# ----------------------------
# Quick inference demonstration (after you trained or load model)
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hindi Meeting Summarizer - train or infer")
    parser.add_argument("--transcript", type=str, required=True, help="Path to Hindi transcript text file")
    parser.add_argument("--train-tokenizer", action="store_true", help="Train new SentencePiece tokenizer on transcript")
    parser.add_argument("--train", action="store_true", help="Train model (uses pseudo summaries if no labels)")
    parser.add_argument("--infer", action="store_true", help="Run summarization + semantic extraction on transcript")
    parser.add_argument("--model-path", type=str, default="", help="Path to saved model .pth for inference")
    args = parser.parse_args()

    # 1) ensure transcript exists
    transcript_file = args.transcript
    if not os.path.exists(transcript_file):
        raise FileNotFoundError("Transcript file not found: " + transcript_file)

    # Optionally train tokenizer & model
    if args.train:
        model, sp = main_train_flow(transcript_file, train_new_tokenizer=args.train_tokenizer)
        # Save tokenizer model file is in hindi_spm.model (if train_new_tokenizer), otherwise use existing TOKENIZER_MODEL
        sp_save_path = TOKENIZER_MODEL if not args.train_tokenizer else "hindi_spm.model"
        print("Tokenizer model:", sp_save_path)
        print("Model trained and available in current dir.")
    else:
        # load tokenizer
        if args.train_tokenizer:
            train_sentencepiece(transcript_file, model_prefix="hindi_spm", vocab_size=VOCAB_SIZE)
            sp_path = "hindi_spm.model"
        else:
            sp_path = TOKENIZER_MODEL
        if not os.path.exists(sp_path):
            raise FileNotFoundError("SentencePiece model not found. Use --train-tokenizer to create one.")
        sp = load_tokenizer(sp_path)

    # Inference
    if args.infer:
        # either load provided model or assume you've trained a model and the latest "transformer_epochX.pth" exists
        if args.model_path:
            assert os.path.exists(args.model_path), "Model file not found: " + args.model_path
            model = load_model(args.model_path, sp.get_piece_size(), sp.get_piece_size()).to(DEVICE)
        else:
            # find latest transformer_epoch*.pth
            pths = sorted([p for p in os.listdir('.') if p.startswith("transformer_epoch") and p.endswith(".pth")])
            if not pths:
                raise FileNotFoundError("No model .pth found. Train first or provide --model-path.")
            model = load_model(pths[-1], sp.get_piece_size(), sp.get_piece_size()).to(DEVICE)

        with open(transcript_file, "r", encoding="utf-8") as f:
            text = f.read()

        print("\n=== Generating summary ===")
        summary = summarize(model, text, sp, max_len=120)
        print("\nSUMMARY (Hindi):\n", summary)

        print("\n=== Extracting semantic insights ===")
        reps, keywords = extract_semantic_insights(text, sp, model, n_topics=4, top_n_sentences=4)
        print("\nRepresentative sentences (topics):")
        for s in reps:
            print("-", s)
        print("\nTop keywords (TF-IDF):")
        print(", ".join(keywords[:30]))
        
    print("\nDone.")
