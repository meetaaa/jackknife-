"""
susenas_preprocessing.py
========================
Praproses data Susenas Kor sesuai Subbab 4.1 (bab hasil):
eksplorasi -> validasi domain -> penanganan skip pattern -> penghapusan baris
(non-respons) -> encoding -> (standardisasi ditangani di dalam pipeline SVR).

Menyediakan:
  - load_metadata(meta_path)
  - clean_raw(df, meta): validasi domain + listwise deletion (kode 8/9)
  - SusenasEncoder: encoding biner 0/1 + one-hot, konsisten antara data
    empiris dan data simulasi (kategori & kolom yang dibuang dipelajari
    dari data empiris).
"""
import numpy as np
import pandas as pd

RESPONSE = "EXPEND"
NONRESPONSE_CODES = (8, 9)  # "tidak tahu" / "menolak menjawab"
FOOD_BLOCK = [f"R170{i}" for i in range(1, 9)]  # R1701..R1708
DK_VAR = ["R1812"]
NOMINAL_VARS = ["R1816", "R2101A", "R1802", "R1803", "R2202"]


def load_metadata(meta_path):
    labels = pd.read_excel(meta_path, sheet_name="labels", header=0)
    labels.columns = ["No", "Kode", "Keterangan", "Satuan", "JumlahKategori"]
    vl = pd.read_excel(meta_path, sheet_name="value labels Kor RT", header=None).iloc[2:].copy()
    vl.columns = ["Var", "Value", "Label"]
    vl["Var"] = vl["Var"].ffill()
    vl = vl.dropna(subset=["Value"])
    vl["Value"] = vl["Value"].astype(int)
    domain = {v: sorted(g["Value"].tolist()) for v, g in vl.groupby("Var")}
    cat = labels.loc[labels["Satuan"].str.strip().str.lower() == "kategorik", "Kode"].tolist()
    num = labels.loc[labels["Satuan"].str.strip().str.lower() == "numerik", "Kode"].tolist()
    return {"domain": domain, "cat": cat, "num": num, "labels": labels}


def clean_raw(df, meta, verbose=True):
    """Validasi domain + penghapusan baris non-respons (listwise deletion).
    Kode 0 pada R1803 (skip pattern) DIPERTAHANKAN sebagai kategori."""
    df = df.copy()
    domain = meta["domain"]

    # --- validasi domain (informatif) ---
    if verbose:
        oob = {}
        for v in meta["cat"]:
            extra = set(df[v].dropna().unique()) - set(domain[v]) - ({0} if v == "R1803" else set())
            if extra:
                oob[v] = sorted(extra)
        n_missing = int(df.isna().sum().sum())
        print(f"[validasi] sel kosong = {n_missing}; nilai di luar domain = {oob if oob else 'tidak ada'}")

    # --- penghapusan baris non-respons (kode 8/9) ---
    drop_mask = df[FOOD_BLOCK + DK_VAR].isin(NONRESPONSE_CODES).any(axis=1)
    n0 = len(df)
    df = df.loc[~drop_mask].reset_index(drop=True)
    if verbose:
        print(f"[hapus baris] {n0} -> {len(df)} (dihapus {int(drop_mask.sum())} baris non-respons)")
    return df


class SusenasEncoder:
    """Encoding konsisten: numerik (di depan) -> biner 0/1 -> R105 -> one-hot.
    Kategori one-hot & kolom konstan yang dibuang dipelajari saat fit
    pada data empiris, lalu diterapkan sama untuk data simulasi."""

    def __init__(self, meta):
        self.meta = meta
        d = meta["domain"]
        self.num = meta["num"]  # numerik (5)
        self.bin_vars = [v for v in meta["cat"] if d[v] in ([1, 5], [1, 5, 8], [1, 5, 8, 9])]
        self.urban = [v for v in meta["cat"] if d[v] == [1, 2]]  # R105
        self.nominal = [v for v in NOMINAL_VARS if v in meta["cat"]]
        self.nominal_cats = {}
        self.feature_names_ = None
        self.numeric_idx_ = None
        self.drop_cols_ = []

    def fit(self, clean_df):
        for v in self.nominal:
            self.nominal_cats[v] = sorted(clean_df[v].unique().tolist())
        X = self._transform_raw(clean_df)
        # kolom konstan (varians nol) dipelajari dari data empiris -> selalu dibuang
        self.drop_cols_ = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=self.drop_cols_)
        self.feature_names_ = X.columns.tolist()
        self.numeric_idx_ = [self.feature_names_.index(v) for v in self.num]
        return self

    def _transform_raw(self, df):
        cols = {}
        for v in self.num:                       # numerik (dijaga di depan)
            cols[v] = df[v].astype(float).values
        for v in self.bin_vars:                  # biner Ya/Tidak -> 1/0
            cols[v] = (df[v].values == 1).astype(int)
        for v in self.urban:                     # R105 -> 1 jika perkotaan
            cols[v] = (df[v].values == 1).astype(int)
        X = pd.DataFrame(cols, index=df.index)
        for v in self.nominal:                   # one-hot (kategori tetap)
            for cat in self.nominal_cats[v]:
                X[f"{v}_{int(cat)}"] = (df[v].values == cat).astype(int)
        return X

    def transform(self, df):
        X = self._transform_raw(df)
        for c in self.drop_cols_:
            if c in X.columns:
                X = X.drop(columns=c)
        # pastikan urutan & kelengkapan kolom sama seperti saat fit
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        return X.values

    def fit_transform(self, clean_df):
        self.fit(clean_df)
        return self.transform(clean_df)
