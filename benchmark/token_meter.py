"""Uniform OpenAI token meter for the cost benchmark (C).

Both pipeline arms run on o3-mini and call exactly two OpenAI endpoints —
``chat.completions.create`` (llm_api + llm_client) and ``beta.chat.completions.parse``
(structured outputs in both). Metering at those two SDK methods therefore measures the rigid
baseline and the agentic arm IDENTICALLY, with no edits to either pipeline's code — so a
cost A/B is apples-to-apples by construction (same instrument, same units).

The agentic arm's own client (`llm_client.py`) discards `response.usage` entirely, and the
rigid arm's `llm_api` accounting uses its own conventions (cached/non-cached split, count_tokens
fallbacks); using one external meter sidesteps both and avoids any methodology mismatch.

Usage::

    with TokenMeter() as m:
        ...run pipeline...
    m.snapshot()  # {"tokens_in", "tokens_out", "calls"}

Scope/caveats: OpenAI-only (the benchmark model); if another provider is used the meter records
nothing for it (snapshot stays 0 — flagged, not silently wrong). Not re-entrant / not
thread-safe — intended for serial per-run use. Degrades to a no-op if the OpenAI SDK or key is
unavailable (snapshot stays 0 and `metered=False`).
"""

from __future__ import annotations


class TokenMeter:
    def __init__(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0
        self.metered = False
        self._orig: dict = {}

    # -- tally a single response's usage (chat.completions + parse share the shape) --
    def _record(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        self.tokens_in += getattr(u, "prompt_tokens", 0) or 0
        self.tokens_out += getattr(u, "completion_tokens", 0) or 0
        self.calls += 1

    def _patch(self, cls, method: str) -> None:
        orig = getattr(cls, method)
        self._orig[(cls, method)] = orig
        meter = self

        def wrapper(inner_self, *args, **kwargs):
            resp = orig(inner_self, *args, **kwargs)
            try:
                meter._record(resp)
            except Exception:  # noqa: BLE001 — never let metering break a real call
                pass
            return resp

        setattr(cls, method, wrapper)

    def __enter__(self) -> TokenMeter:
        try:
            import openai

            client = openai.OpenAI()
            chat_completions_cls = type(client.chat.completions)  # has .create
            beta_completions_cls = type(client.beta.chat.completions)  # has .parse
            self._patch(chat_completions_cls, "create")
            self._patch(beta_completions_cls, "parse")
            self.metered = True
        except Exception:  # noqa: BLE001 — SDK/key absent → no-op meter
            self.metered = False
        return self

    def __exit__(self, *exc) -> bool:
        for (cls, method), orig in self._orig.items():
            setattr(cls, method, orig)
        self._orig.clear()
        return False

    def snapshot(self) -> dict:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "calls": self.calls,
            "metered": self.metered,
        }
