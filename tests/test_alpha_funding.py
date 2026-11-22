"""Safety tests for the standalone funding observation lane."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_funding import publisher, signal


def test_quantile_basics():
    assert signal.quantile([1, 2, 3, 4], 0.0) == 1
    assert signal.quantile([1, 2, 3, 4], 1.0) == 4
    assert signal.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert signal.quantile([7], 0.15) == 7


def test_evaluate_fires_at_or_below_percentile():
    points = [(i, 0.01 * i) for i in range(100)]
    points[-1] = (100, -0.5)  # latest funding collapses below the lower band
    r = signal.evaluate(points, q_lo=0.15, min_history=20)
    assert r["fire"] is True
    assert r["direction"] == "SHORT"
    assert r["history_n"] == 99
    assert r["threshold"] < 0.01 * 99


def test_evaluate_does_not_fire_when_high():
    points = [(i, 0.01 * i) for i in range(100)]
    points[-1] = (100, 1.0)
    r = signal.evaluate(points, q_lo=0.15, min_history=20)
    assert r["fire"] is False
    assert r["direction"] is None


def test_evaluate_requires_history():
    r = signal.evaluate([(1, 0.1)], q_lo=0.15, min_history=20)
    assert r["fire"] is False
    assert r["reason"] == "insufficient_history"


def test_threshold_never_uses_latest_observation():
    points = [(i, 0.0) for i in range(50)] + [(50, -9.0)]
    r = signal.evaluate(points, q_lo=0.15, min_history=20)
    assert r["threshold"] == 0.0, "latest rate leaked into the threshold"


def test_executor_symbol():
    assert signal.executor_symbol("BTC/USDT") == "BTCUSDT"
    assert signal.executor_symbol("XAU/USDT") == "XAUUSDT"


def test_payload_is_inert_shadow():
    result = {"fire": True, "reason": "x", "symbol_ts": 1,
              "funding_rate": -0.001, "threshold": 0.0,
              "percentile": 0.15, "history_n": 90, "direction": "SHORT",
              "evaluated_at": "t"}
    p = publisher.payload_for("BTC/USDT", result)
    assert p["mode"] == "shadow"
    assert p["rsi_lower"] == 0.0
    assert p["rsi_upper"] == 100.0
    assert p["direction"] == "SHORT"
    assert p["research"]["status"] == "shadow"
    assert p["research"]["signal"]["funding_rate"] == -0.001
    # 0/100 are outside CanArm's strict (0,100) interval -> never arms
    assert not (0.0 < p["rsi_upper"] < 100.0)


def test_payload_no_fire_is_long_inert():
    result = {"fire": False, "reason": "y", "symbol_ts": 1,
              "funding_rate": 0.01, "threshold": 0.0,
              "percentile": 0.15, "history_n": 90, "direction": None,
              "evaluated_at": "t"}
    p = publisher.payload_for("ETH/USDT", result)
    assert p["direction"] == "LONG"
    assert p["rsi_lower"] == 0.0 and p["rsi_upper"] == 100.0


def test_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    publisher.save_state({"BTCUSDT": {"symbol_ts": 123}}, str(path))
    assert publisher.load_state(str(path))["BTCUSDT"]["symbol_ts"] == 123


def test_audit_append(tmp_path):
    path = tmp_path / "audit.jsonl"
    publisher.append_audit([{"a": 1}, {"a": 2}], str(path))
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(x) for x in fh]
    assert [x["a"] for x in lines] == [1, 2]


def test_module_imports_do_not_touch_main():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "alpha_funding", "signal.py"), encoding="utf-8").read()
    assert "from main import" not in src
    assert "import config\n" not in src.replace("from . import config", "")
