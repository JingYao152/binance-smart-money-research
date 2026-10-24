import json

import pytest

from test_llm import FakeClient, bundle, payload
from smartmoney_research.cli import main
from smartmoney_research.llm import generate_hypothesis, render_skill


def test_skill_contains_citations_labels_and_research_controls(bundle, payload):
    text = render_skill(generate_hypothesis(bundle, FakeClient(payload)), bundle)
    for token in [*bundle.evidence_ids, "observed", "derived", "hypothesis", "account_equity",
                  "sample_count", "partial", "ema_crossover", "1x", "2x", "scale-in",
                  "fee_bps", "slippage_bps", "drawdown", "2024-07-01", "Untested"]:
        assert token in text


def test_skill_cli_writes_local_outputs(tmp_path, monkeypatch, bundle, payload):
    monkeypatch.setattr("smartmoney_research.llm.LLMClient.complete", lambda self, evidence: payload)
    monkeypatch.setattr("smartmoney_research.analytics.evidence.build_evidence_bundle", lambda store, trader: bundle)
    output = tmp_path / "generated_skills" / "synthetic"
    assert main(["skill", "--database", str(tmp_path / "test.duckdb"),
                 "--trader-id", bundle.trader_id, "--output", str(output)]) == 0
    assert "hypothesis" in (output / "SKILL.md").read_text()
    assert json.loads((output / "evidence.json").read_text()) == bundle.to_dict()
    assert (output / ".gitignore").read_text() == "*\n"


def test_skill_cli_fails_without_key_before_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    output = tmp_path / "generated_skills"
    assert main(["skill", "--database", str(tmp_path / "test.duckdb"),
                 "--trader-id", "synthetic", "--output", str(output)]) == 2
    assert "LLM_API_KEY" in capsys.readouterr().err
    assert not output.exists()


def test_skill_rejects_changed_bundle(bundle, payload):
    hypothesis = generate_hypothesis(bundle, FakeClient(payload))
    bundle.claim_candidates[0]["sample_count"] = 1
    with pytest.raises(ValueError, match="bundle"):
        render_skill(hypothesis, bundle)


def test_skill_output_does_not_overwrite_existing_files(tmp_path, bundle, payload):
    from smartmoney_research.llm import write_skill
    public_template = tmp_path / "SKILL.md"
    public_template.write_text("generic public template")
    hypothesis = generate_hypothesis(bundle, FakeClient(payload))
    with pytest.raises(FileExistsError):
        write_skill(tmp_path, hypothesis, bundle)
    assert public_template.read_text() == "generic public template"
    assert not (tmp_path / "evidence.json").exists()


def test_skill_output_rejects_symlink_directory(tmp_path, bundle, payload):
    from smartmoney_research.llm import write_skill
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "output"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        write_skill(link, generate_hypothesis(bundle, FakeClient(payload)), bundle)
    assert not list(target.iterdir())


def test_output_does_not_hide_existing_unrelated_files(tmp_path, bundle, payload):
    from smartmoney_research.llm import write_skill
    (tmp_path / "source.py").write_text("source content")
    with pytest.raises(FileExistsError):
        write_skill(tmp_path, generate_hypothesis(bundle, FakeClient(payload)), bundle)
    assert not (tmp_path / ".gitignore").exists()


def test_source_markdown_is_inert_in_every_dynamic_field(bundle, payload):
    from dataclasses import replace
    import re

    hostile = '![image](https://example.invalid/image)\n```\n[link](https://example.invalid/link)\n``````'
    hostile_id = "position:" + hostile
    payload["claims"][0]["evidence_ids"] = [hostile_id, "position:2"]
    bundle.evidence_ids.remove("position:1")
    bundle.evidence_ids.add(hostile_id)
    bundle.items[hostile_id] = bundle.items.pop("position:1")
    bundle.claim_candidates[0]["evidence_ids"] = [hostile_id, "position:2"]
    candidate_value = {"symbol": hostile, "value": "[metric](https://example.invalid/metric)"}
    bundle.claim_candidates[0]["value"] = candidate_value
    bundle = replace(bundle, trader_id=hostile, coverage_status=hostile,
                     coverage={hostile: {"symbol": hostile}}, sample_counts={hostile: 2},
                     missing_fields={hostile: [hostile]})

    text = render_skill(generate_hypothesis(bundle, FakeClient(payload)), bundle)
    # Parse fenced blocks using Markdown's closing-fence rules. JSON round-trips
    # prove source values were preserved without allowing any link/image markup
    # to escape into prose, including strings containing newlines and backticks.
    fence = None
    block = []
    decoded_blocks = []
    prose = []
    for line in text.splitlines():
        if fence is None:
            opening = re.fullmatch(r"(`{3,})json", line)
            if opening:
                fence = opening[1]
                block = []
            else:
                prose.append(line)
        elif re.fullmatch(r"`{" + str(len(fence)) + r",}\s*", line):
            decoded_blocks.append(json.loads("\n".join(block)))
            fence = None
        else:
            block.append(line)
    assert fence is None
    assert hostile in decoded_blocks
    assert bundle.coverage in decoded_blocks
    assert bundle.sample_counts in decoded_blocks
    assert bundle.missing_fields in decoded_blocks
    assert [candidate_value] in decoded_blocks
    assert payload["claims"][0] in decoded_blocks
    assert sorted(bundle.evidence_ids) in decoded_blocks
    assert "https://example.invalid" not in "\n".join(prose)
    assert not any(re.search(r"!?\[.*\]\(", line) for line in prose)
