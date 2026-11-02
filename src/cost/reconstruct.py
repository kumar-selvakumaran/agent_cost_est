"""Turn accounting and reference cost (spec §5.2–5.3)."""
from __future__ import annotations

import numpy as np

ASSISTANT_ROLES = {"assistant", "gpt"}
ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system", "user": "user",
            "assistant": "assistant", "tool": "tool"}


def normalize_messages(conv) -> list[tuple[str, str]]:
    """Accept ShareGPT (from/value) or OpenAI (role/content) messages -> [(role, content)]."""
    out = []
    for m in conv:
        if "role" in m and m.get("role") is not None:
            r, c = m.get("role"), m.get("content")
        else:
            r, c = m.get("from"), m.get("value")
        out.append((ROLE_MAP.get(r, r), c if isinstance(c, str) else ""))
    return out


def turn_accounting(tok: np.ndarray, is_asst: np.ndarray, c_msg: int = 4) -> dict:
    """tok: content tokens per message (without overhead); is_asst: bool mask.

    Returns per-turn arrays I, O, Cch, U and observation length obs (tokens of the
    non-assistant messages between turn k and turn k+1, incl. overhead).
    """
    t = np.asarray(tok, dtype=np.int64) + c_msg
    is_asst = np.asarray(is_asst, dtype=bool)
    a = np.flatnonzero(is_asst)
    csum = np.concatenate([[0], np.cumsum(t)])
    I = csum[a]                       # sum of tok(m_i) for i < a_k
    O = t[a]
    Cch = np.zeros_like(I)
    if len(a) > 1:
        Cch[1:] = np.minimum(I[:-1] + O[:-1], I[1:])
    U = I - Cch
    nxt = np.append(a[1:], len(t))
    obs = csum[nxt] - csum[a + 1]
    return dict(I=I, O=O, Cch=Cch, U=U, obs=obs)


def run_cost(acc: dict, lam_c: float = 0.1, lam_o: float = 4.0) -> np.ndarray:
    """Per-turn reference cost contributions."""
    return acc["U"] + lam_c * acc["Cch"] + lam_o * acc["O"]


def targets(acc: dict, cfg_targets: dict | None = None) -> dict:
    cfg_targets = cfg_targets or {}
    lc, lo = cfg_targets.get("lambda_c", 0.1), cfg_targets.get("lambda_o", 4.0)
    lhi = cfg_targets.get("lambda_o_hi", 8.0)
    Y = float(np.sum(run_cost(acc, lc, lo)))
    Y_tok = float(np.sum(acc["I"] + acc["O"]))
    Y_hi = float(np.sum(run_cost(acc, lc, lhi)))
    return dict(Y=Y, Y_tok=Y_tok, Y_hi=Y_hi, K=int(len(acc["I"])))
