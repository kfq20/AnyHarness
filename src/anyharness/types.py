"""The ``Sample`` dataclass, slimmed from Slime's ``slime.utils.types.Sample``.

Only the fields and the ``Status`` enum that the trajectory layer needs are kept;
the multimodal, top-p replay, and routed-experts machinery (and its torch /
``slime.utils.misc`` dependencies) is dropped because this project is text-only
and reads from /v1/messages captures, not an SGLang rollout engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Sample:
    """The sample generated."""

    group_index: int | None = None
    index: int | None = None
    # Id of the rollout this sample came from. Defaults to ``None`` and the
    # downstream pipeline falls back to ``index`` (so the default rollout
    # path, where one execution = one training sample, sees rollout_id ==
    # index). Compact / subagent paths that split one rollout execution into
    # multiple training samples should set the same ``rollout_id`` on every
    # sibling, so loss aggregation averages within the rollout instead of
    # over-counting it.
    rollout_id: int | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    # response
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    # top-k alternative logprobs per output token: a typed TopkLogprobs (paired
    # token_ids + logprobs matrices, shape (num_tokens, k), sentinel-padded).
    # None when SLIME_TOP_LOGPROBS not set; enables nucleus-replay / offline resampling RL.
    output_top_logprobs: "TopkLogprobs | None" = None  # type: ignore[name-defined]
    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Sample":
        data = dict(data)
        data["status"] = Sample.Status(data["status"])

        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    @property
    def effective_response_length(self) -> int:
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length
