"""
Simple tokenizer and dataset for RLM demo

Provides:
- SimpleTokenizer: whitespace + punctuation tokenizer with small vocab builder
- RLMDataset: torch Dataset that returns token ids and float targets
"""
from typing import List, Iterable, Optional, Tuple
import re
from collections import Counter
import torch
from torch.utils.data import Dataset


class SimpleTokenizer:
    def __init__(self, unk_token: str = "<UNK>", pad_token: str = "<PAD>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.vocab = {pad_token: 0, unk_token: 1}
        self.inv_vocab = {0: pad_token, 1: unk_token}
        self._fitted = False

    def fit(self, texts: Iterable[str], min_freq: int = 1, max_vocab: Optional[int] = None):
        tokens = []
        for t in texts:
            tokens.extend(self._tokenize(t))

        counts = Counter(tokens)
        items = [w for w, c in counts.most_common() if c >= min_freq]
        if max_vocab:
            items = items[: max_vocab - len(self.vocab)]

        # start idx at next available integer (use inv_vocab keys which are ints)
        next_idx = max(self.inv_vocab.keys()) + 1
        for w in items:
            if w not in self.vocab:
                self.vocab[w] = next_idx
                self.inv_vocab[next_idx] = w
                next_idx += 1

        self._fitted = True

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # simple tokenization: split on non-word characters
        tokens = [t for t in re.split(r"\W+", text) if t]
        return tokens

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]
        if max_len is not None:
            if len(ids) < max_len:
                ids = ids + [self.vocab[self.pad_token]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        return ids

    def decode_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.inv_vocab.get(i, self.unk_token) for i in ids]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class RLMDataset(Dataset):
    def __init__(self, samples: Iterable[Tuple[str, float]], tokenizer: SimpleTokenizer, max_len: int = 128):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # If tokenizer not fitted, fit on provided texts
        if not getattr(self.tokenizer, "_fitted", False):
            texts = [s[0] for s in self.samples]
            try:
                self.tokenizer.fit(texts)
            except Exception:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, target = self.samples[idx]
        input_ids = self.tokenizer.encode(text, max_len=self.max_len)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target = torch.tensor(float(target), dtype=torch.float32)
        return input_ids, target
