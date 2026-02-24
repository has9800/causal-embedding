from __future__ import annotations

import os
import re
from typing import Any, Dict, Tuple


class ClaudeCritic:
    """Anthropic scoring client with resilient fallback parser."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg["critic"].get("use_premium_critic", False))
        self._client = None
        if self.enabled:
            from anthropic import Anthropic

            api_key = os.getenv(cfg["critic"]["anthropic_api_key_env"], "")
            if not api_key:
                raise RuntimeError("Premium critic enabled but ANTHROPIC_API_KEY is missing.")
            self._client = Anthropic(api_key=api_key)

    def score_trace(self, prompt: str, trace: str) -> Tuple[float, str]:
        if not self.enabled:
            return 0.0, "premium critic disabled"

        system_prompt = (
            "You are a strict causal reasoning evaluator. Score the trace from 0 to 10 "
            "for causal validity (not style), then provide one-sentence justification. "
            "Output format: SCORE: <number>\\nJUSTIFICATION: <text>."
        )
        msg = self._client.messages.create(
            model=self.cfg["critic"]["anthropic_model"],
            max_tokens=120,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"PROMPT:\n{prompt}\n\nTRACE:\n{trace}",
                }
            ],
        )
        text = msg.content[0].text if msg.content else ""
        return self._parse_score(text), text

    @staticmethod
    def _parse_score(raw: str) -> float:
        match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
        if not match:
            return 0.0
        return max(0.0, min(10.0, float(match.group(1))))
