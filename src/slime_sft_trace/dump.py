"""Dump SFT training data from a list of :class:`Sample`.

Writes:
* ``trajectories.jsonl`` — one JSON record per sample (prompt, response, tokens,
  loss_mask, rollout_log_probs, metadata).
* ``trajectories.md`` — human-readable prompt + response per sample.
* ``trajectories.pt`` (optional) — a ``torch.save``-d dict, only when torch is
  importable.

The JSONL schema mirrors what the trajectory layer produces: ``tokens`` is the
full token sequence (prompt + response), ``loss_mask`` is aligned to the
response region (prompt stripped, response=1, appended prompt-tails=0), and
``response`` is the decoded response text the adapter fills in.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _sample_to_record(s: Any) -> dict[str, Any]:
    """Flatten one Sample to the JSONL record schema."""
    return {
        "prompt": s.prompt,
        "response": s.response,
        "tokens": s.tokens,
        "loss_mask": s.loss_mask,
        "rollout_log_probs": s.rollout_log_probs,
        "metadata": s.metadata,
    }


def dump_samples(samples: list[Any], out_dir: str | Path) -> Path:
    """Write SFT data for ``samples`` under ``out_dir`` and return the dir.

    Always writes ``trajectories.jsonl`` and ``trajectories.md``. Writes
    ``trajectories.pt`` too when torch is available. Creates ``out_dir`` if
    needed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- trajectories.jsonl ---
    jsonl_path = out / "trajectories.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(_sample_to_record(s), ensure_ascii=False) + "\n")

    # --- trajectories.md (human-readable) ---
    md_path = out / "trajectories.md"
    lines: list[str] = [f"# Trajectories ({len(samples)} sample(s))\n"]
    for i, s in enumerate(samples):
        prompt = s.prompt
        if isinstance(prompt, list):
            prompt = json.dumps(prompt, ensure_ascii=False, indent=2)
        response = s.response or ""
        n_tok = len(s.tokens) if s.tokens else 0
        n_resp = sum(s.loss_mask) if s.loss_mask else 0
        lines.append(f"\n---\n\n## Sample {i}\n")
        lines.append(f"_tokens={n_tok}, trained={n_resp}, reward={s.reward}_\n")
        lines.append(f"\n### Prompt\n\n```\n{prompt}\n```\n")
        lines.append(f"\n### Response\n\n```\n{response}\n```\n")
    md_path.write_text("".join(lines), encoding="utf-8")

    # --- trajectories.pt (optional, only if torch importable) ---
    try:
        import torch  # noqa: F401  (import guard)
    except Exception:
        logger.info("torch not importable; skipping trajectories.pt")
    else:
        pt_path = out / "trajectories.pt"
        records = [_sample_to_record(s) for s in samples]
        torch.save({"samples": records}, pt_path)  # type: ignore[name-defined]
        logger.info("wrote %d samples to %s", len(samples), pt_path)

    logger.info("dumped %d sample(s) to %s", len(samples), out)
    return out


def _sample_to_sft_record(s: Any) -> dict[str, Any]:
    """Flatten one Sample to the message-level SFT JSONL record schema.

    OpenAI/HF messages + a separate ``tools`` column:
    * ``messages`` -> ``s.prompt`` when it is a messages list, else a single
      user message wrapping the prompt string.
    * ``tools`` -> ``s.metadata['tools']`` when present, else omitted.
    * ``meta``  -> session_id / reward / use_tool / truncated.
    No mask is written here; trainers derive it via ``assistant_only_loss``.

    The tool-call ``arguments`` stay as a dict here (the form the production
    GLM-5.2 converter emits for its primary file). The ``openai_wire`` flavor
    stringifies them in :func:`_stringify_tool_arguments` (mirroring the
    production converter's ``stringify_tool_arguments``), because
    ``datasets<4.7.0`` infers ``tool_calls.function.arguments`` as a string and
    ``tools`` as a string unless JSON-stringified -- the dict form needs
    ``datasets>=4.7`` or an explicit ``Json()`` Features schema.
    """
    prompt = s.prompt
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    md = s.metadata or {}
    record: dict[str, Any] = {"messages": messages}
    if "tools" in md and md["tools"] is not None:
        record["tools"] = md["tools"]
    record["meta"] = {
        "session_id": md.get("session_id"),
        "reward": s.reward,
        "use_tool": md.get("use_tool", False),
        "truncated": md.get("truncated", False),
    }
    return record


# Format strings mirror the production converter
# (export_covered_claude_events_to_glm52_sft.py): the primary GLM-5.2 file keeps
# ``tool_call.function.arguments`` as a dict, the OpenAI-wire companion
# stringifies them. ``datasets<4.7.0`` only loads the dict form correctly when
# arguments is already a JSON string (it infers ``str`` for both
# ``tool_calls.function.arguments`` and ``tools``); the dict form needs
# ``datasets>=4.7`` or an explicit ``datasets.Features`` ``Json()`` schema.
_SFT_FORMATS = {
    "glm52": "glm52_chat_messages_with_tools_jsonl",
    "openai_wire": "openai_chat_messages_with_tools_jsonl",
}

# Filename per flavor: the default (glm52) keeps the bare name existing callers
# read; openai_wire writes a distinct companion file so both flavors can coexist
# in the same out_dir.
_SFT_FILENAMES = {
    "glm52": "trajectories_sft.jsonl",
    "openai_wire": "trajectories_sft_openai.jsonl",
}


def _stringify_tool_arguments(record: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``record`` with every
    ``messages[].tool_calls[].function.arguments`` JSON-stringified and
    ``meta.format`` set to the OpenAI-wire marker.

    Faithful copy of the production converter's ``stringify_tool_arguments``
    (``export_covered_claude_events_to_glm52_sft.py``): it deepcopies, sets
    ``meta["format"] = "openai_chat_messages_with_tools_jsonl"``, then for every
    tool call stringifies ``function.arguments`` (which is a dict in the base
    record) with ``json.dumps(args or {}, ensure_ascii=False)``. Already-string
    arguments are left untouched.
    """
    converted = copy.deepcopy(record)
    converted["meta"]["format"] = _SFT_FORMATS["openai_wire"]
    for message in converted.get("messages") or []:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                fn["arguments"] = json.dumps(args or {}, ensure_ascii=False)
    return converted


def dump_samples_sft(
    samples: list[Any], out_dir: str | Path, *, flavor: str = "glm52"
) -> Path:
    """Write message-level SFT trajectories, in one of two portable flavors.

    One JSON record per Sample (OpenAI/HF ``messages`` + optional ``tools`` +
    ``meta``). Creates ``out_dir`` if needed. Returns the output directory.

    ``flavor`` selects the on-disk representation of tool-call arguments so
    users can match their ``datasets`` version:

    * ``"glm52"`` (default): ``tool_calls[].function.arguments`` stay a **dict**
      and ``meta.format = "glm52_chat_messages_with_tools_jsonl"``. This is the
      form the production GLM-5.2 converter emits for its primary file; it needs
      ``datasets>=4.7`` (or an explicit ``Json()`` Features schema) to load
      correctly. Written to ``trajectories_sft.jsonl`` (the name existing
      callers read).
    * ``"openai_wire"``: every ``tool_calls[].function.arguments`` is
      JSON-stringified (``json.dumps(args, ensure_ascii=False)``) and
      ``meta.format = "openai_chat_messages_with_tools_jsonl"``. This is the
      portable form: ``datasets<4.7`` infers ``str`` for both
      ``tool_calls.function.arguments`` and ``tools``, so the stringified form
      loads without a Features schema. Written to
      ``trajectories_sft_openai.jsonl``.

    The stringify + format-marker logic mirrors the production converter's
    ``stringify_tool_arguments`` exactly.
    """
    if flavor not in _SFT_FORMATS:
        raise ValueError(
            f"flavor must be 'glm52' or 'openai_wire', got {flavor!r}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / _SFT_FILENAMES[flavor]
    with jsonl_path.open("w", encoding="utf-8") as f:
        for s in samples:
            record = _sample_to_sft_record(s)
            record["meta"]["format"] = _SFT_FORMATS[flavor]
            if flavor == "openai_wire":
                record = _stringify_tool_arguments(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "dumped %d message-level sample(s) (flavor=%s) to %s",
        len(samples),
        flavor,
        jsonl_path,
    )
    return out


__all__ = ["dump_samples", "dump_samples_sft", "_stringify_tool_arguments"]
