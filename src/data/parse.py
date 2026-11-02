"""Parse one terminus-2 transcript into per-turn accounting and prefix signals (spec §5, §6.3)."""
from __future__ import annotations

import json
import re

import numpy as np

from src.cost.reconstruct import normalize_messages, turn_accounting, targets
from src.data.audit import extract_task_text, TASK_MARK

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
ERR_PATTERNS = ["Traceback", "Error", "error:", "command not found", "No such file", "FAILED", "Permission denied"]
HANDOFF_RE = re.compile(r"(picking up work from a previous|other agent provided|previous (?:ai )?agent)", re.I)


def parse_assistant(content: str):
    """Return (keystrokes tuple, task_complete flag, parse_failed flag)."""
    s = THINK_RE.sub("", content)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return (), False, True
    try:
        obj = json.loads(s[i:j + 1])
    except Exception:
        tc = bool(re.search(r'"task_complete"\s*:\s*true', s))
        ks = tuple(re.findall(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
        return ks, tc, True
    if not isinstance(obj, dict):
        return (), False, True
    cmds = obj.get("commands") or []
    ks = []
    if isinstance(cmds, list):
        for c in cmds:
            if isinstance(c, dict):
                ks.append(str(c.get("keystrokes", "")))
            else:
                ks.append(str(c))
    tc = obj.get("task_complete") in (True, "true", "True")
    return tuple(ks), bool(tc), False


def parse_run(conv, tok_fn, c_msg: int = 4, lam=None):
    """conv: raw list of message dicts. tok_fn(list[str]) -> np.ndarray of token counts.

    Returns (run_dict, turns_dict_of_arrays).
    """
    msgs = normalize_messages(conv)
    roles = [r for r, _ in msgs]
    contents = [c for _, c in msgs]
    tok = tok_fn(contents)
    is_a = np.array([r == "assistant" for r in roles])
    acc = turn_accounting(tok, is_a, c_msg)
    tg = targets(acc, lam)
    a_idx = np.flatnonzero(is_a)
    K = len(a_idx)
    lc = (lam or {}).get("lambda_c", 0.1)
    lo = (lam or {}).get("lambda_o", 4.0)
    cost_k = acc["U"] + lc * acc["Cch"] + lo * acc["O"]
    # observation text following each assistant turn
    err = np.zeros(K, dtype=np.int8)
    rep = np.zeros(K, dtype=np.int8)
    empty = np.zeros(K, dtype=np.int8)
    tcf = np.zeros(K, dtype=np.int8)
    pfail = np.zeros(K, dtype=np.int8)
    handoff = np.zeros(K, dtype=np.int8)
    seen = set()
    for k, ai in enumerate(a_idx):
        ks, tc, pf = parse_assistant(contents[ai])
        empty[k] = len([x for x in ks if x.strip()]) == 0
        key = tuple(x.strip() for x in ks)
        rep[k] = (not empty[k]) and key in seen
        if not empty[k]:
            seen.add(key)
        tcf[k] = tc
        pfail[k] = pf
        end = a_idx[k + 1] if k + 1 < K else len(msgs)
        obs_text = "\n".join(contents[ai + 1:end])
        err[k] = any(p in obs_text for p in ERR_PATTERNS)
        handoff[k] = bool(HANDOFF_RE.search(obs_text[:600])) if obs_text else False
    first_user = next((c for r, c in msgs if r == "user"), "")
    task_text = extract_task_text(first_user)
    i_task = first_user.rfind(TASK_MARK)
    harness = first_user[:i_task] if i_task >= 0 else ""
    ht = tok_fn([harness, task_text])
    n_think = int(sum(1 for ai in a_idx if "<think>" in contents[ai] or "</think>" in contents[ai]))
    sys_tok = int(sum(tok[i] + c_msg for i, r in enumerate(roles) if r == "system")) + int(ht[0])
    run = dict(tg)
    run.update(
        task_text=task_text, instr_tokens=int(ht[1]), sys_tokens=sys_tok, n_msgs=len(msgs),
        n_think=n_think, think_share=n_think / max(K, 1),
        n_handoff=int(handoff.sum()), final_task_complete=bool(tcf[-1]) if K else False,
        any_task_complete=bool(tcf.any()) if K else False,
        parse_fail_share=float(pfail.mean()) if K else 0.0,
        msg_prefix_hashes=None,
    )
    turns = dict(k=np.arange(1, K + 1), I=acc["I"], O=acc["O"], Cch=acc["Cch"], U=acc["U"], obs=acc["obs"],
                 cost=cost_k, cum_cost=np.cumsum(cost_k), err=err, rep=rep, empty=empty, tc=tcf,
                 pfail=pfail, handoff=handoff)
    return run, turns
