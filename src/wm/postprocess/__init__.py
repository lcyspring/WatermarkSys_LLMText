"""后处理子包：随机删词、截断、同义替换（语义改写/回译为选做扩展）。"""
from wm.postprocess.delete import EditResult, delete_words
from wm.postprocess.synonym import SynonymResult, replace_synonyms
from wm.postprocess.truncate import TruncateResult, truncate

__all__ = [
    "delete_words", "EditResult",
    "truncate", "TruncateResult",
    "replace_synonyms", "SynonymResult",
]
