"""Experiment-5 constants and helpers."""

import re
from typing import Optional


EXPERIMENT5_DEFAULT_OBJECT_PROMPT = (
    "milk. cereal. cornflakes. oatmeal. "
    "bottle. can. box. bag. tool. cup. mug. bowl. jar. carton."
)

EXPERIMENT5_TARGET_KEYWORDS = {
    "Milk": ["milk"],
    "Cereal": ["cereal", "cornflakes", "oatmeal", "granola"],
}


def normalize_text_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_label_to_experiment5_target(label: str) -> Optional[str]:
    norm = normalize_text_label(label)
    for target_name, keywords in EXPERIMENT5_TARGET_KEYWORDS.items():
        for kw in keywords:
            if kw in norm:
                return target_name
    return None
