"""后处理子包：字符串级与 token 级两种实现。"""
from wm.postprocess.delete import EditResult, delete_words
from wm.postprocess.synonym import SynonymResult, replace_synonyms
from wm.postprocess.token_edits import (
    EDIT_LEVELS,
    TokenEditResult,
    apply_edit,
    delete_random,
    substitute_random,
    truncate as truncate_tokens,
)
from wm.postprocess.truncate import TruncateResult, truncate

__all__ = [
    "delete_words", "EditResult",
    "replace_synonyms", "SynonymResult",
    "TokenEditResult", "delete_random", "substitute_random",
    "truncate", "TruncateResult", "truncate_tokens",
    "apply_edit", "EDIT_LEVELS",
]
