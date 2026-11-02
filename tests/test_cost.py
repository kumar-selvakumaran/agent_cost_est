import numpy as np
import pytest

from src.cost.reconstruct import normalize_messages, turn_accounting, targets
from src.cost.tokenizers import count_tokens


def test_three_turn_hand_computed():
    # messages: user(10) asst(5) user(20) asst(3) user(7) asst(2); C_MSG = 4
    tok = np.array([10, 5, 20, 3, 7, 2])
    is_a = np.array([0, 1, 0, 1, 0, 1], bool)
    acc = turn_accounting(tok, is_a, c_msg=4)
    t = tok + 4  # 14 9 24 7 11 6
    assert list(acc["I"]) == [14, 14 + 9 + 24, 14 + 9 + 24 + 7 + 11]
    assert list(acc["O"]) == [9, 7, 6]
    assert list(acc["Cch"]) == [0, 14 + 9, 47 + 7]
    assert list(acc["U"]) == [14, 24, 11]
    assert list(acc["obs"]) == [24, 11, 0]
    Y = (14 + 24 + 11) + 0.1 * (0 + 23 + 54) + 4 * (9 + 7 + 6)
    tg = targets(acc)
    assert tg["Y"] == pytest.approx(Y)
    assert tg["Y_tok"] == pytest.approx(sum([14, 47, 65]) + 22)
    assert tg["Y_hi"] == pytest.approx((14 + 24 + 11) + 0.1 * 77 + 8 * 22)
    assert tg["K"] == 3


def test_identities_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 40)
        tok = rng.integers(0, 5000, n)
        is_a = rng.random(n) < 0.5
        acc = turn_accounting(tok, is_a)
        assert acc["U"].sum() + acc["Cch"].sum() == acc["I"].sum()
        assert (acc["U"] >= 0).all()
        assert (acc["Cch"] <= acc["I"]).all()


def test_sharegpt_vs_openai_same_Y():
    content = ["You are an agent. Task: fix bug", '{"analysis": "x", "commands": []}', "output\n$ ls", '{"task_complete": true}']
    oa = [{"role": r, "content": c} for r, c in zip(["user", "assistant", "user", "assistant"], content)]
    sg = [{"from": r, "value": c} for r, c in zip(["human", "gpt", "human", "gpt"], content)]
    res = []
    for conv in (oa, sg):
        msgs = normalize_messages(conv)
        tok = count_tokens([c for _, c in msgs])
        acc = turn_accounting(tok, np.array([r == "assistant" for r, _ in msgs]))
        res.append(targets(acc)["Y"])
    assert res[0] == res[1]


def test_zero_assistant_turns():
    acc = turn_accounting(np.array([5, 6]), np.array([False, False]))
    assert len(acc["I"]) == 0 and targets(acc)["Y"] == 0
