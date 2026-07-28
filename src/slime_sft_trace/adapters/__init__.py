"""HTTP adapters for agent rollouts (vendored from Slime).

Only the Anthropic Messages adapter is vendored; the OpenAI adapter is dropped.
The upstream call inside BaseAdapter._run_turn is made pluggable via
UPSTREAM_MODE (sglang | messages) — see adapters.common.
"""

from slime_sft_trace.adapters.anthropic import AnthropicAdapter
from slime_sft_trace.adapters.common import BaseAdapter

__all__ = ["AnthropicAdapter", "BaseAdapter"]
