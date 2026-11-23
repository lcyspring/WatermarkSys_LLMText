"""同义替换：对候选内容词按比例替换，记录可替换数与实际替换数。

正式实现可接入 NLTK WordNet（延迟导入）；无 NLTK 时退化为内置小词典，
保证模块可在断网/最小环境下运行与测试。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# 最小内置同义词典（仅用于流程贯通与单测；正式实验须用 WordNet 并记录词典版本）
_BUILTIN_SYNONYMS = {
    "good": ["great", "fine"], "bad": ["poor", "terrible"],
    "big": ["large", "huge"], "small": ["tiny", "little"],
    "happy": ["glad", "pleased"], "said": ["stated", "remarked"],
}

# 只对内容词（名词/动词/形容词/副词）替换；简易白名单词性前缀
CONTENT_POS_PREFIXES = ("N", "V", "J", "R")


@dataclass
class SynonymResult:
    text: str
    tokens: list[str] = field(default_factory=list)
    candidate_count: int = 0
    replaced_count: int = 0
    actual_rate: float = 0.0
    failed: bool = False
    error: str = ""


def _wordnet_synonyms(word: str):
    try:
        from nltk.corpus import wordnet  # 延迟导入
        seen = set()
        out = []
        for syn in wordnet.synsets(word):
            for lem in syn.lemmas():
                name = lem.name().replace("_", " ")
                if name != word and name not in seen:
                    seen.add(name)
                    out.append(name)
        return out
    except Exception:
        return _BUILTIN_SYNONYMS.get(word.lower(), [])


def replace_synonyms(
    text: str,
    rate: float,
    seed: int = 0,
    tokenizer=None,
) -> SynonymResult:
    if tokenizer is not None:
        tokens = tokenizer.tokenize(text)
    else:
        tokens = text.split()
    # 找候选词（有同义词的内容词）
    candidate_idx = []
    for i, tok in enumerate(tokens):
        word = tok.lower().strip(".,!?;:\"'()")
        if _wordnet_synonyms(word):
            candidate_idx.append(i)
    candidate_count = len(candidate_idx)
    if candidate_count == 0:
        out = " ".join(tokens) if tokenizer is None else tokenizer.convert_tokens_to_string(tokens)
        return SynonymResult(out, list(tokens), 0, 0, 0.0)
    n_replace = max(1, int(round(rate * candidate_count)))
    n_replace = min(n_replace, candidate_count)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(candidate_idx, n_replace))
    replaced = 0
    for i in chosen:
        word = tokens[i].lower().strip(".,!?;:\"'()")
        syns = _wordnet_synonyms(word)
        if not syns:
            continue
        tokens[i] = syns[rng.randrange(len(syns))]
        replaced += 1
    out = " ".join(tokens) if tokenizer is None else tokenizer.convert_tokens_to_string(tokens)
    return SynonymResult(out, tokens, candidate_count, replaced,
                         replaced / candidate_count)
