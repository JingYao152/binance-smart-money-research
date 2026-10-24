---
name: trader-research
description: Generic workflow for local evidence-cited strategy research
---

# Evidence-cited trader research

Read only the local evidence bundle. Keep trader-specific output under generated_skills/ or another ignored output directory. Never place source records, credentials, or generated trader-specific Skills in tracked files.

Read LLM_API_KEY from the environment. Optional LLM_BASE_URL selects an OpenAI-compatible API base, and LLM_MODEL selects its model. Generation fails without a key. Do not request source API data during hypothesis generation.

Use the version 1.0 strategy-hypothesis JSON contract. Cite existing evidence IDs with sample counts. Label source rows observed, exact bundle candidate statistics derived, and proposed strategies hypothesis. Treat missing fields as missing. Do not infer trader intent, invent trades, or claim causal relationships or completed backtests.

Only ema_crossover and rsi_threshold are supported. Use one long or short direction per run. Use fully closed bars available before each event, and execute signals no earlier than the following bar. Proposed validation windows must be chronological and disjoint; verify actual data coverage separately.

Default leverage is 1x; the configurable maximum cannot exceed 2x. State sizing, fee and slippage assumptions, funding gaps, exits, drawdown controls, sample counts, coverage and missing-data notes. Scale-in and pyramiding are unsupported. A generated Skill is an untested research hypothesis, and may never execute orders or model-generated Python.

Run smartmoney-research skill --trader-id LOCAL_ID --output generated_skills/LOCAL_ID to write local SKILL.md and evidence.json. Choose a fresh output directory for each run; existing artifacts and symlink paths are rejected. Review those files and run a separate allow-listed backtest before treating any condition as tested.
