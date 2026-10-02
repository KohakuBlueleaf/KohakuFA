"""KohakuFA: flash attention for Blackwell in Gluon, with a numerically careful softmax.

    from kohakufa import attention
    out = attention(q, k, v, causal=True)  # q [B, H, Sq, D], k / v [B, H, Skv, D]

See README.md for what is supported and docs/precision.md for the numerics.
"""

from kohakufa.api import attention

__all__ = ["attention"]
__version__ = "0.1.0"
