"""基于 jieba 分词的 BM25 稀疏向量生成器，输出 Qdrant SparseVector 格式。"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import jieba


class BM25SparseEncoder:
    """对中文语料库拟合 → 每个文档生成稀疏向量（indices + values）。"""
    VERSION = 1

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}   # token → index
        self._idf: dict[str, float] = {}   # token → IDF
        self._avgdl: float = 0.0           # 平均文档长度（token 数）

    # ── 拟合 ──
    def fit(self, texts: list[str]):
        """在语料库上构建词表并计算 IDF。"""
        tokenized = [self._tokenize(t) for t in texts]
        doc_counts = Counter()
        doc_lens = []
        for tokens in tokenized:
            doc_lens.append(len(tokens))
            for t in set(tokens):
                doc_counts[t] += 1

        n = len(texts)
        self._avgdl = sum(doc_lens) / max(n, 1)

        # 按 token 出现次数倒序分配 index（常见词 index 更小）
        sorted_tokens = sorted(doc_counts.items(), key=lambda x: -x[1])
        self._vocab = {t: i for i, (t, _) in enumerate(sorted_tokens)}

        for t, df in doc_counts.items():
            self._idf[t] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    # ── 持久化 ──
    def save(self, path: str | Path):
        p = Path(path)
        data = {
            "version": self.VERSION,
            "k1": self.k1, "b": self.b,
            "avgdl": self._avgdl,
            "vocab": self._vocab,
            "idf": self._idf,
        }
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25SparseEncoder":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        encoder = cls(k1=data["k1"], b=data["b"])
        encoder._avgdl = data["avgdl"]
        encoder._vocab = data["vocab"]
        encoder._idf = data["idf"]
        return encoder

    # ── 编码 ──
    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """将单个文本编码为稀疏向量 (indices, values)。"""
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        tf = Counter(tokens)
        doc_len = len(tokens)

        indices = []
        values = []
        for t, f in tf.items():
            idx = self._vocab.get(t)
            if idx is None:
                continue
            # BM25 文档端权重
            tf_norm = f / (f + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1)))
            weight = self._idf[t] * tf_norm
            indices.append(idx)
            values.append(weight)

        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """将查询编码为稀疏向量，只保留在词汇表中的词。"""
        tokens = self._tokenize(text)
        if not tokens:
            return [], []
        tf = Counter(tokens)
        indices = []
        values = []
        for t, f in tf.items():
            idx = self._vocab.get(t)
            if idx is None:
                continue
            # 查询端：简单的增强 TF
            weight = f * (1 + math.log(max(f, 1)))
            indices.append(idx)
            values.append(weight)
        return indices, values

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # jieba 精确模式 + 过滤单字和纯标点
        words = jieba.lcut(text)
        return [w.strip() for w in words if len(w.strip()) > 1]
