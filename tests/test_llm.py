import copy
import json
from urllib.error import URLError

import pytest

from smartmoney_research.analytics.evidence import EvidenceBundle
from smartmoney_research.llm import (
    HypothesisValidationError, LLMClient, generate_hypothesis, validate_hypothesis,
)


@pytest.fixture
def bundle():
    return EvidenceBundle(
        trader_id="synthetic-trader", coverage_status="partial", incomplete=True,
        coverage={"position_history": 2}, sample_counts={"closed_positions": 2},
        missing_fields={"analysis": ["account_equity"]}, metrics={"win_rate": 0.5},
        evidence_ids={"position:1", "position:2"},
        items={eid: {"label": "observed", "table": "position_history"}
               for eid in ["position:1", "position:2"]},
        claim_candidates=[{"claim": "known_outcome_win_rate", "label": "derived", "value": 0.5,
                           "sample_count": 2, "evidence_ids": ["position:1", "position:2"]}],
    )


@pytest.fixture
def payload():
    return {
        "schema_version": "1.0",
        "claims": [{"label": "derived", "claim": "known_outcome_win_rate",
                    "sample_count": 2, "evidence_ids": ["position:1", "position:2"]}],
        "strategy": {"kind": "ema_crossover", "direction": "long", "fast": 12, "slow": 26},
        "risk": {"leverage": 1, "max_leverage": 2, "position_fraction": 0.1,
                 "max_drawdown": 0.2, "fee_bps": 5, "slippage_bps": 2},
        "validation": {"train_start": "2024-01-01", "train_end": "2024-06-30",
                       "test_start": "2024-07-01", "test_end": "2024-12-31"},
    }


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, bundle):
        return copy.deepcopy(self.payload)


def test_valid_response_is_bound_to_local_evidence(bundle, payload):
    hypothesis = generate_hypothesis(bundle, FakeClient(payload))
    assert hypothesis.strategy.kind == "ema_crossover"
    assert hypothesis.to_dict()["claims"] == payload["claims"]


@pytest.mark.parametrize("mutation", [
    lambda p: p["claims"][0].update(evidence_ids=["invented"]),
    lambda p: p["strategy"].update(kind="execute_python"),
    lambda p: p["claims"][0].pop("sample_count"),
    lambda p: p["claims"][0].update(claim="Trader intends to pump the market"),
    lambda p: p["claims"][0].update(sample_count=True),
    lambda p: p["claims"][0].update(evidence_ids=[]),
    lambda p: p["claims"][0].update(label="fact"),
    lambda p: p["strategy"].update(code="print('unsafe')"),
    lambda p: p["risk"].update(leverage=3),
    lambda p: p["risk"].update(max_leverage=3),
    lambda p: p["validation"].update(test_start="2024-01-01"),
    lambda p: p.update(extra="unsupported"),
])
def test_rejects_unsupported_or_unverifiable_output(bundle, payload, mutation):
    mutation(payload)
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis(payload, bundle.evidence_ids)


def test_rejects_invented_sample_count(bundle, payload):
    payload["claims"][0]["sample_count"] = 50
    with pytest.raises(HypothesisValidationError, match="bundle"):
        generate_hypothesis(bundle, FakeClient(payload))


def test_rejects_wrong_candidate_citations(bundle, payload):
    payload["claims"][0]["evidence_ids"] = ["position:1"]
    with pytest.raises(HypothesisValidationError, match="bundle"):
        generate_hypothesis(bundle, FakeClient(payload))


def test_no_http_without_key(monkeypatch, bundle):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("network accessed without key")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        LLMClient().complete(bundle)


def test_openai_compatible_request_uses_only_bundle(monkeypatch, bundle, payload):
    monkeypatch.setenv("LLM_API_KEY", "synthetic-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "synthetic-model")
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode()
    def respond(request, timeout):
        assert request.full_url == "https://llm.example.invalid/v1/chat/completions"
        body = json.loads(request.data)
        assert json.loads(body["messages"][1]["content"]) == bundle.to_dict()
        assert body["model"] == "synthetic-model"
        assert "synthetic-secret" not in request.data.decode()
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", respond)
    assert generate_hypothesis(bundle, LLMClient()).strategy.kind == "ema_crossover"


def test_transport_error_does_not_echo_credentials(monkeypatch, bundle):
    monkeypatch.setenv("LLM_API_KEY", "synthetic-secret")
    def fail(*args, **kwargs):
        raise URLError("synthetic-secret")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError, match="LLM request failed") as exc:
        LLMClient().complete(bundle)
    assert "synthetic-secret" not in str(exc.value)


def test_empty_claims_are_rejected(bundle, payload):
    payload["claims"] = []
    with pytest.raises(HypothesisValidationError):
        generate_hypothesis(bundle, FakeClient(payload))


def test_default_leverage_is_one(bundle, payload):
    del payload["risk"]["leverage"]
    assert generate_hypothesis(bundle, FakeClient(payload)).risk["leverage"] == 1


def test_observed_and_hypothesis_claims_count_only_local_records(bundle, payload):
    payload["claims"] = [
        {"label": "observed", "claim": "source_record", "evidence_ids": ["position:1"], "sample_count": 1},
        {"label": "hypothesis", "claim": "strategy_candidate", "evidence_ids": ["position:1", "position:2"], "sample_count": 2},
    ]
    assert len(generate_hypothesis(bundle, FakeClient(payload)).claims) == 2
    bundle.items["position:1"]["table"] = "raw_responses"
    with pytest.raises(HypothesisValidationError):
        generate_hypothesis(bundle, FakeClient(payload))
