"""slime_sft_trace — the HiLR (Harness-in-the-Loop) layer.

HiLR is the layer between a harness (Claude Code) and the sampling endpoint /
API server: the :class:`~slime_sft_trace.adapters.AnthropicAdapter` exposes
``/v1/messages`` downstream to the harness, forwards upstream to a pluggable
backend (sglang with per-token logprobs, or any messages-API), and captures
each turn into a per-session trajectory tree that an offline aggregator dumps
as SFT data. "Harness in the loop" = the adapter sits *between* the harness
and the model, intercepting every ``/v1/messages`` call rather than
fire-and-forget.

Vendors Slime's :class:`TrajectoryManager` + :class:`TurnRecord` (and the
supporting :class:`MessageNode`, :class:`_SampleBuilder`, :class:`DriftKind`)
together with the slimmed :class:`Sample` dataclass. Only the trajectory +
types layer lives here; the adapter and the harness+CLI are handled elsewhere.
"""

from __future__ import annotations

from .trajectory import DriftKind, MessageNode, TrajectoryManager, TurnRecord
from .types import Sample

__all__ = [
    "TrajectoryManager",
    "TurnRecord",
    "MessageNode",
    "Sample",
    "DriftKind",
]
