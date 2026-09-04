"""
Provider-agnostic LLM client abstraction for the PhIDO pipeline.

Supports OpenAI, Anthropic (Claude), and Google (Gemini) with a unified
interface for tool-calling loops and structured (Pydantic) output.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

_log = logging.getLogger(__name__)


class EmptyResponseError(RuntimeError):
    """A 2xx completion that carries no ``choices``.

    OpenRouter (notably the ``:free`` tiers) answers HTTP 200 with ``choices: null``
    and an ``error`` body when the upstream provider fails or sheds load.  The SDK
    raises nothing, so an unguarded ``response.choices[0]`` surfaces as a bare
    ``TypeError: 'NoneType' object is not subscriptable`` — indistinguishable from a
    genuine model failure when it lands in a benchmark record.  We raise this
    instead: it is retried like any transient provider error, and if it survives
    the retries the caller records an honest ``error:EmptyResponseError``.
    """


def _require_choices(resp: Any, *, what: str) -> Any:
    """Return ``resp`` if it carries choices; otherwise raise :class:`EmptyResponseError`.

    Placed *inside* the retried callable so a null-choices 200 re-enters the
    backoff loop rather than escaping as a TypeError downstream.
    """
    if getattr(resp, "choices", None):
        return resp
    err = getattr(resp, "error", None)
    if err is None:
        extra = getattr(resp, "model_extra", None)
        if isinstance(extra, dict):
            err = extra.get("error")
    raise EmptyResponseError(
        f"{what}: provider returned no choices (HTTP 2xx); error={str(err)[:300]}"
    )


def _require_content(resp: Any, *, what: str) -> Any:
    """Return ``resp`` if choices[0] carries non-empty content; else raise EmptyResponseError.

    Companion to :func:`_require_choices` for calls where empty content is never a valid
    answer (JSON-mode structured output, no tools). Under load-shed OpenRouter also
    degrades to 2xx responses whose message content is empty/None — feeding those to the
    validation-repair loop burns its attempts on provider noise and surfaces as a bogus
    'model failure'. Same contract: placed inside the retried callable so it re-enters
    the patient shed backoff.
    """
    content = resp.choices[0].message.content
    if content and content.strip():
        return resp
    fr = getattr(resp.choices[0], "finish_reason", None)
    raise EmptyResponseError(
        f"{what}: provider returned empty content (HTTP 2xx, finish_reason={fr})"
    )


def _is_context_length_error(e: Exception) -> bool:
    """True if a ``BadRequestError`` is a context-window overflow.

    Context overflow is *deterministic* — the accumulated agentic tool-call +
    critic history simply exceeds the model's window — so retrying it is pure
    wasted latency (we observed ~516s burned across 4 backoff attempts).  OpenAI
    signals it via ``code == "context_length_exceeded"`` in the error body, with
    the human-readable message also containing "maximum context length".  We
    check both the structured code and the message text so a body-shape change
    doesn't silently reopen the retry loop.
    """
    code = getattr(e, "code", None)
    if code == "context_length_exceeded":
        return True
    body = getattr(e, "body", None)
    if isinstance(body, dict) and body.get("code") == "context_length_exceeded":
        return True
    msg = str(e).lower()
    return "context_length_exceeded" in msg or "maximum context length" in msg


def _openai_with_retries(fn, *, what: str, max_attempts: int = 4):
    """Call an OpenAI SDK function with bounded exponential backoff.

    Retries clearly-transient errors (rate limit, timeout, connection, 5xx, and a
    2xx-with-no-choices provider shed — see :class:`EmptyResponseError`) — the
    unambiguous case. ``BadRequestError`` (400) is usually a malformed request, but has
    been observed INTERMITTENTLY on o3-mini selection completions (the same code path
    succeeds on other prompts / other runs), which points at content-dependent
    serialization or a flaky server response. So we retry it a bounded couple of times
    AND log the body + request shape for diagnosis, rather than silently swallowing it.

    Exception: a ``context_length_exceeded`` 400 is deterministic (the history is too
    big) — retrying can never help, so we fail fast and let the caller handle it.
    """
    from openai import (
        APIConnectionError,
        APITimeoutError,
        BadRequestError,
        InternalServerError,
        RateLimitError,
    )
    transient = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    # Provider load-shed (a 2xx with no choices — free tiers under demand) gets its own,
    # far more patient schedule: shed windows last minutes, not seconds, so the standard
    # 4-attempt/16s backoff would fail runs that a patient retry rides out.
    shed_max = int(os.getenv("EMPTY_RESPONSE_MAX_ATTEMPTS", "10"))
    attempt, shed = 0, 0
    while True:
        try:
            return fn()
        except BadRequestError as e:
            if _is_context_length_error(e):
                _log.warning("OpenAI 400 context_length_exceeded on %s — failing fast "
                             "(no retry): %s", what, str(e)[:500])
                raise
            attempt += 1
            _log.warning("OpenAI 400 on %s (attempt %d/%d): %s",
                         what, attempt, max_attempts, str(e)[:500])
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 ** attempt, 16))
        except EmptyResponseError:
            shed += 1
            _log.warning("provider load-shed (2xx, no choices) on %s (shed retry %d/%d)",
                         what, shed, shed_max)
            if shed >= shed_max:
                raise
            time.sleep(min(15 * shed, 120))
        except transient as e:
            attempt += 1
            _log.warning("OpenAI transient %s on %s (attempt %d/%d)",
                         type(e).__name__, what, attempt, max_attempts)
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 ** attempt, 16))

# ---------------------------------------------------------------------------
# Normalized response types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Provider-agnostic representation of a single tool/function call."""
    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass
class LLMResponse:
    """Normalized response from any provider."""
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"  # "stop", "tool_calls", "length"
    raw: Any = None  # provider's native message object for appending to history


# ---------------------------------------------------------------------------
# Tool-definition converters (OpenAI format -> provider format)
# ---------------------------------------------------------------------------

def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI function-calling tool defs to Anthropic format."""
    converted = []
    for tool in tools:
        fn = tool.get("function", tool)
        params = dict(fn.get("parameters", {}))
        params.pop("additionalProperties", None)
        converted.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": params,
        })
    return converted


def _openai_tools_to_gemini(tools: list[dict]):
    """Convert OpenAI function-calling tool defs to Gemini FunctionDeclaration list."""
    import google.generativeai as genai

    def _clean_schema(d):
        """Strip JSON Schema keywords unsupported by Gemini.

        Only removes ``default`` and ``title`` from *property definition*
        dicts (those that contain a ``type`` key), not from ``properties``
        container dicts where ``default`` or ``title`` may be legitimate
        property names.
        """
        if isinstance(d, dict):
            d.pop("additionalProperties", None)
            if "type" in d:
                d.pop("default", None)
                d.pop("title", None)
            for v in d.values():
                _clean_schema(v)
            # Reconcile required vs. properties so no required entry
            # references a property that doesn't exist.
            if "required" in d and "properties" in d:
                d["required"] = [
                    r for r in d["required"] if r in d["properties"]
                ]
                if not d["required"]:
                    del d["required"]
        elif isinstance(d, list):
            for v in d:
                _clean_schema(v)

    declarations = []
    for tool in tools:
        fn = tool.get("function", tool)
        params = dict(fn.get("parameters", {}))
        _clean_schema(params)
        declarations.append(genai.types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=params if params.get("properties") else None,
        ))
    return [genai.types.Tool(function_declarations=declarations)]


# ---------------------------------------------------------------------------
# Structured-output helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(txt: str) -> str:
    txt = (txt or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", txt, re.IGNORECASE)
    return m.group(1).strip() if m else txt


def _extract_balanced_json(txt: str) -> str:
    s = _strip_code_fences(txt)
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return s


def _parse_structured(raw_text: str, response_model: type[BaseModel]) -> BaseModel:
    """Parse raw JSON text into a Pydantic model with fallback extraction."""
    try:
        return response_model.model_validate_json(raw_text)
    except Exception:
        return response_model.model_validate_json(_extract_balanced_json(raw_text))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Provider-agnostic LLM client."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(
        self,
        messages: list,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request, optionally with tool definitions."""

    @abstractmethod
    def complete_structured(
        self,
        messages: list,
        response_model: type[BaseModel],
        system: str | None = None,
    ) -> BaseModel:
        """Request structured output conforming to a Pydantic model."""

    @abstractmethod
    def assistant_message(self, response: LLMResponse) -> Any:
        """Build the provider-specific assistant message to append to history."""

    @abstractmethod
    def tool_result_message(self, tool_call: ToolCall, result: str) -> dict:
        """Build the provider-specific tool-result message to append to history."""

    @property
    def supports_vision(self) -> bool:
        """Whether this provider/model can accept image inputs."""
        return False

    def complete_vision(
        self,
        text: str,
        image_bytes: bytes,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a vision completion with a text prompt and a single PNG image.

        Subclasses that support vision override this to build the
        provider-specific multimodal message and call the API.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support vision input."
        )


# ---------------------------------------------------------------------------
# Token-usage accounting (the agentic client otherwise discards response.usage).
# Process-global running tally; read/reset by callers (e.g. the cost-estimate pilot).
# ---------------------------------------------------------------------------

USAGE = {"in": 0, "out": 0, "calls": 0, "cached_in": 0, "cost": 0.0}


def _tally(resp: Any, provider: str) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    if provider == "openai":
        USAGE["in"] += getattr(u, "prompt_tokens", 0) or 0
        USAGE["out"] += getattr(u, "completion_tokens", 0) or 0
        # cached prefix tokens (billed at 10% of input) — subset of prompt_tokens
        det = getattr(u, "prompt_tokens_details", None)
        USAGE["cached_in"] += (getattr(det, "cached_tokens", 0) or 0) if det else 0
        # OpenRouter returns the actual billed cost (USD) in usage.cost when
        # extra_body usage.include=True; native OpenAI omits it (stays 0).
        USAGE["cost"] += float(getattr(u, "cost", 0) or 0)
    else:  # anthropic: input/output + cache reads count as input at read price
        USAGE["in"] += (getattr(u, "input_tokens", 0) or 0) + \
            (getattr(u, "cache_read_input_tokens", 0) or 0) + \
            (getattr(u, "cache_creation_input_tokens", 0) or 0)
        USAGE["out"] += getattr(u, "output_tokens", 0) or 0
    USAGE["calls"] += 1


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-run trace collector (module-level; serial use only — the driver snapshots
# and clears it around each run). Every step is best-effort and never raises.
# ---------------------------------------------------------------------------
_TRACE_BUFFER: list[dict] = []


def trace_reset() -> None:
    _TRACE_BUFFER.clear()


def trace_dump() -> list[dict]:
    return list(_TRACE_BUFFER)


def _trace_add(step: dict) -> None:
    try:
        _TRACE_BUFFER.append(step)
    except Exception:  # noqa: BLE001 — tracing must never break a run
        pass


class OpenAIClient(LLMClient):

    # Most OpenAI chat models accept image_url content, but the text-only
    # reasoning-mini models reject it ("image_url is only supported by certain
    # models"), so vision support must be decided per-model, not per-provider.
    _NO_VISION_PREFIXES = ("o1-mini", "o3-mini")

    @property
    def supports_vision(self) -> bool:
        return not self.model.startswith(self._NO_VISION_PREFIXES)

    def __init__(self, model: str):
        super().__init__(model)
        from openai import OpenAI
        if "/" in model:  # OpenRouter slug, e.g. "qwen/qwen3.6-27b"
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            # enable the model's reasoning/thinking on OpenRouter, and ask
            # OpenRouter to return the actual billed cost in usage.cost
            self._extra_body = {"reasoning": {"enabled": True}, "usage": {"include": True}}
            # Optional provider pin (comma-separated OpenRouter provider names), e.g. to keep a
            # paid-route refill on the provider whose behaviour matches the original run.
            _order = [p.strip() for p in os.getenv("OPENROUTER_PROVIDER_ORDER", "").split(",") if p.strip()]
            if _order:
                self._extra_body["provider"] = {"order": _order, "allow_fallbacks": True}
        else:
            self._client = OpenAI()
            self._extra_body = {}

    def complete(self, messages, system=None, tools=None) -> LLMResponse:
        msgs = list(messages)
        if system and not any(
            (isinstance(m, dict) and m.get("role") == "system") for m in msgs
        ):
            msgs.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {"model": self.model, "messages": msgs}
        if tools:
            kwargs["tools"] = tools
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        if os.getenv("PHIDO_DEBUG_TOKENS"):
            _log.warning("DEBUG complete model=%s n_msgs=%d ~chars=%d",
                         self.model, len(msgs), sum(len(str(m)) for m in msgs))

        response = _openai_with_retries(
            lambda: _require_choices(
                self._client.chat.completions.create(**kwargs), what="complete"),
            what="complete")
        _tally(response, "openai")
        choice = response.choices[0]

        tc_list = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tc_list.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        stop = "tool_calls" if tc_list else (
            "length" if choice.finish_reason == "length" else "stop"
        )

        _trace_add({
            "kind": "complete",
            "tool_calls": [{"name": tc.name, "args": (tc.arguments or "")[:600]} for tc in tc_list],
            "content": (choice.message.content or "")[:400],
            "reasoning": (getattr(choice.message, "reasoning", None) or "")[:1200],
            "stop": stop,
        })

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tc_list,
            stop_reason=stop,
            raw=choice.message,
        )

    def complete_structured(self, messages, response_model, system=None):
        msgs = list(messages)
        if system and not any(
            (isinstance(m, dict) and m.get("role") == "system") for m in msgs
        ):
            msgs.insert(0, {"role": "system", "content": system})

        if os.getenv("PHIDO_DEBUG_TOKENS"):
            _log.warning("DEBUG complete_structured model=%s schema=%s n_msgs=%d ~chars=%d",
                         self.model, getattr(response_model, "__name__", "?"),
                         len(msgs), sum(len(str(m)) for m in msgs))

        # OpenRouter/open models often do not honour OpenAI strict-schema `.parse`
        # (they drop required fields). For those, drive JSON mode with the schema in the
        # prompt and a self-repair loop that feeds validation errors back to the model.
        if "/" in self.model:
            return self._complete_structured_json(msgs, response_model)

        response = _openai_with_retries(
            lambda: _require_choices(self._client.beta.chat.completions.parse(
                model=self.model,
                messages=msgs,
                response_format=response_model,
            ), what="complete_structured"), what="complete_structured")
        _tally(response, "openai")
        msg = response.choices[0].message
        if msg.parsed:
            return msg.parsed
        raise ValueError(f"Structured output parsing failed: {msg.refusal or 'no parsed result'}")

    def _complete_structured_json(self, msgs, response_model, max_attempts: int = 3):
        """Schema-in-prompt JSON mode with a validation-repair loop (for OpenRouter/open models)."""
        import json as _json
        schema = _json.dumps(response_model.model_json_schema())
        instr = ("Respond with ONLY a single JSON object that conforms exactly to this JSON Schema. "
                 "Include every required field; where you have no value use an empty array or object, "
                 "but never omit a required field.\n\nSchema:\n" + schema)
        call_msgs = list(msgs) + [{"role": "user", "content": instr}]
        last_err = None
        last_fr, last_len = None, 0
        for _ in range(max_attempts):
            response = _openai_with_retries(
                lambda: _require_content(_require_choices(self._client.chat.completions.create(
                    model=self.model, messages=call_msgs,
                    # PHIDO_JSON_MODE=0 omits response_format: providers that honour JSON mode
                    # (e.g. Venice) suppress the model's reasoning under it, and OpenRouter will
                    # not route a response_format request to providers lacking the parameter.
                    **({} if os.getenv("PHIDO_JSON_MODE", "1") == "0" else {"response_format": {"type": "json_object"}}),
                    extra_body=self._extra_body or None,
                ), what="complete_structured_json"), what="complete_structured_json"),
                what="complete_structured_json")
            _tally(response, "openai")
            content = response.choices[0].message.content or ""
            last_fr = getattr(response.choices[0], "finish_reason", None)
            last_len = len(content)
            try:
                parsed = response_model.model_validate_json(content)
                _trace_add({"kind": "structured",
                            "schema": getattr(response_model, "__name__", "?"),
                            "content": content[:400]})
                return parsed
            except Exception as e:  # noqa: BLE001 — feed the error back and retry
                last_err = e
                call_msgs = call_msgs + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content":
                        f"That JSON failed schema validation: {str(e)[:400]}. "
                        "Return corrected JSON with ALL required fields present."},
                ]
        raise ValueError(
            f"OpenRouter structured output failed after {max_attempts} attempts "
            f"(last: finish_reason={last_fr}, content_len={last_len}): {last_err}")

    def assistant_message(self, response: LLMResponse) -> Any:
        return response.raw

    def tool_result_message(self, tool_call: ToolCall, result: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        }

    def complete_vision(self, text, image_bytes, system=None) -> LLMResponse:
        import base64
        b64 = base64.b64encode(image_bytes).decode()
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                }},
            ],
        })
        response = _openai_with_retries(
            lambda: _require_choices(self._client.chat.completions.create(
                model=self.model, messages=msgs,
                extra_body=self._extra_body or None,
            ), what="complete_vision"), what="complete_vision")
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            stop_reason="stop",
            raw=choice.message,
        )


# ---------------------------------------------------------------------------
# Anthropic (Claude) implementation
# ---------------------------------------------------------------------------

def _claude_code_oauth_token() -> str:
    """Return Claude Code's current subscription OAuth access token.

    Read fresh from ~/.claude/.credentials.json on each call so a token Claude Code refreshes
    mid-run is picked up. The access token is short-lived (``expiresAt``); for runs that outlast
    it, Claude Code (or a refresh step) must rewrite the file — a standalone long run with no
    refresher will start 401ing once ``expiresAt`` passes.
    """
    import pathlib
    path = pathlib.Path.home() / ".claude" / ".credentials.json"
    data = json.loads(path.read_text(encoding="utf-8"))["claudeAiOauth"]
    return data["accessToken"]


class AnthropicClient(LLMClient):

    supports_vision = True

    def __init__(self, model: str):
        super().__init__(model)
        import anthropic
        # PHIDO_CLAUDE_CODE_AUTH=1 -> authenticate with Claude Code's own subscription OAuth
        # credential (~/.claude/.credentials.json) instead of an ANTHROPIC_API_KEY. The token is
        # read fresh here (per-client construction, i.e. per pipeline call) so any refresh Claude
        # Code writes to the file is picked up; the oauth beta header is required for /v1/messages
        # with a Bearer OAuth token. Retries are bumped because the subscription tier throttles
        # (429) far more aggressively than an API key. Falls back to the API-key path when off.
        if os.getenv("PHIDO_CLAUDE_CODE_AUTH") == "1":
            self._client = anthropic.Anthropic(
                auth_token=_claude_code_oauth_token(),
                default_headers={"anthropic-beta": "oauth-2025-04-20"},
                max_retries=int(os.getenv("PHIDO_ANTHROPIC_MAX_RETRIES", "8")),
            )
        else:
            self._client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        self._tool_cache: list[dict] | None = None

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        if tools is None:
            return None
        if self._tool_cache is None:
            self._tool_cache = _openai_tools_to_anthropic(tools)
        return self._tool_cache

    @staticmethod
    def _split_system(messages: list, system: str | None):
        """Extract system content; Anthropic takes it as a top-level param."""
        sys_parts = []
        if system:
            sys_parts.append(system)
        filtered = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                sys_parts.append(m["content"])
            else:
                filtered.append(m)
        return "\n\n".join(sys_parts) if sys_parts else None, filtered

    def complete(self, messages, system=None, tools=None) -> LLMResponse:
        sys_text, msgs = self._split_system(messages, system)
        converted_tools = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16384,
            "messages": msgs,
        }
        if sys_text:
            kwargs["system"] = sys_text
        if converted_tools:
            kwargs["tools"] = converted_tools

        response = self._client.messages.create(**kwargs)
        _tally(response, "anthropic")

        content_text = ""
        tc_list = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tc_list.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                ))

        stop = "tool_calls" if tc_list else (
            "length" if response.stop_reason == "max_tokens" else "stop"
        )

        return LLMResponse(
            content=content_text or None,
            tool_calls=tc_list,
            stop_reason=stop,
            raw=response,
        )

    @staticmethod
    def _resolve_refs(schema: dict) -> dict:
        """Inline all ``$ref`` pointers so the schema contains no ``$defs``.

        Anthropic's tool ``input_schema`` does not support JSON Schema
        ``$ref``.  This resolves every ``{"$ref": "#/$defs/Foo"}`` in-place
        by substituting the referenced definition, then drops ``$defs``.
        Handles recursive / circular refs by capping depth.
        """
        defs = schema.get("$defs", {})
        if not defs:
            return schema

        def _resolve(node, depth=0):
            if depth > 20:
                return node
            if isinstance(node, dict):
                if "$ref" in node:
                    ref_path = node["$ref"]  # e.g. "#/$defs/ComponentIntent"
                    ref_name = ref_path.rsplit("/", 1)[-1]
                    resolved = defs.get(ref_name, node)
                    return _resolve(dict(resolved), depth + 1)
                return {k: _resolve(v, depth) for k, v in node.items()}
            if isinstance(node, list):
                return [_resolve(item, depth) for item in node]
            return node

        resolved = _resolve(schema)
        resolved.pop("$defs", None)
        return resolved

    @staticmethod
    def _condense_messages(messages: list) -> list[dict]:
        """Flatten a conversation with tool_use/tool_result blocks into plain text.

        The Anthropic API rejects ``tool_choice`` if the history contains
        ``tool_use`` blocks referencing tools not in the current ``tools``
        list.  This extracts text from all blocks and merges consecutive
        same-role messages so the result is a clean user/assistant sequence.
        """
        flat: list[dict] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Extract text from Anthropic content blocks
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            parts.append(f"[called {block.get('name', '?')}]")
                        elif block.get("type") == "tool_result":
                            inner = block.get("content", "")
                            if isinstance(inner, str):
                                parts.append(f"[tool result: {inner[:500]}]")
                    elif hasattr(block, "text"):
                        parts.append(block.text)
                text = "\n".join(p for p in parts if p)
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)

            if not text.strip():
                continue

            # Map roles to user/assistant only
            if role in ("tool", "function"):
                role = "user"
            elif role == "system":
                continue  # system handled separately
            elif role not in ("user", "assistant"):
                role = "user"

            # Merge consecutive same-role messages
            if flat and flat[-1]["role"] == role:
                flat[-1]["content"] += "\n\n" + text
            else:
                flat.append({"role": role, "content": text})

        # Anthropic requires the conversation to start with a user message
        if flat and flat[0]["role"] != "user":
            flat.insert(0, {"role": "user", "content": "Begin."})

        # Anthropic requires the conversation to end with a user message
        if flat and flat[-1]["role"] != "user":
            flat.append({"role": "user", "content": "Now produce the requested output."})

        return flat

    def complete_structured(self, messages, response_model, system=None):
        """Force structured output via Anthropic's forced tool_choice.

        Condenses the conversation history to plain text (stripping tool_use /
        tool_result blocks), resolves ``$ref`` in the Pydantic JSON Schema,
        then uses ``tool_choice`` to force structured output.
        """
        schema = response_model.model_json_schema()

        def _clean(d):
            if isinstance(d, dict):
                d.pop("title", None)
                d.pop("default", None)
                for v in d.values():
                    _clean(v)
            elif isinstance(d, list):
                for v in d:
                    _clean(v)
        _clean(schema)

        schema = self._resolve_refs(schema)

        sys_text, raw_msgs = self._split_system(messages, system)
        msgs = self._condense_messages(raw_msgs)

        tool_name = "structured_output"
        tool_def = {
            "name": tool_name,
            "description": (
                f"Return a valid {response_model.__name__} object.  "
                "You MUST call this tool with the complete structured data."
            ),
            "input_schema": schema,
        }

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16384,
            "messages": msgs,
            "tools": [tool_def],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        if sys_text:
            kwargs["system"] = sys_text

        response = self._client.messages.create(**kwargs)
        _tally(response, "anthropic")

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                raw_json = (
                    json.dumps(block.input)
                    if isinstance(block.input, dict)
                    else str(block.input)
                )
                return _parse_structured(raw_json, response_model)

        # If no tool_use block found (shouldn't happen with tool_choice),
        # try to extract JSON from any text blocks
        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text += block.text
        if raw_text.strip():
            return _parse_structured(raw_text, response_model)

        raise ValueError(
            f"Anthropic returned no tool_use block and no parseable text "
            f"for {response_model.__name__}"
        )

    def assistant_message(self, response: LLMResponse) -> dict:
        blocks = []
        if response.content:
            blocks.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            args = json.loads(tc.arguments) if tc.arguments else {}
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": args,
            })
        return {"role": "assistant", "content": blocks}

    def tool_result_message(self, tool_call: ToolCall, result: str) -> dict:
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result,
            }],
        }

    def complete_vision(self, text, image_bytes, system=None) -> LLMResponse:
        import base64
        b64 = base64.b64encode(image_bytes).decode()
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                }},
                {"type": "text", "text": text},
            ],
        }]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        content_text = ""
        for block in response.content:
            if block.type == "text":
                content_text += block.text
        return LLMResponse(
            content=content_text or None,
            stop_reason="stop",
            raw=response,
        )


# ---------------------------------------------------------------------------
# Google Gemini implementation
# ---------------------------------------------------------------------------

class GeminiClient(LLMClient):

    supports_vision = True

    def __init__(self, model: str):
        super().__init__(model)
        import google.generativeai as genai
        self._genai = genai

        api_key = (
            os.getenv("GOOGLEGENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        genai.configure(api_key=api_key)
        self._tool_cache = None

    def _convert_tools(self, tools: list[dict] | None):
        if tools is None:
            return None
        if self._tool_cache is None:
            self._tool_cache = _openai_tools_to_gemini(tools)
        return self._tool_cache

    @staticmethod
    def _to_gemini_messages(messages: list, system: str | None):
        """Convert messages to Gemini contents + system_instruction.

        Messages produced by ``assistant_message`` / ``tool_result_message``
        already contain ``genai.protos.Part`` objects in their ``parts``
        list — these are passed through as-is.  Plain-text messages (e.g.
        the initial system/user messages) are wrapped in a text Part.

        Consecutive same-role messages are merged (Gemini rejects adjacent
        messages with the same role).
        """
        import google.generativeai as genai

        genai_contents: list[dict] = []
        sys_parts: list[str] = []
        if system:
            sys_parts.append(system)

        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "user")

            if role == "system":
                sys_parts.append(m.get("content", ""))
                continue

            if role in ("assistant", "model"):
                gemini_role = "model"
            elif role == "tool":
                gemini_role = "user"
            else:
                gemini_role = "user"

            # Reuse pre-built protos.Part objects when present;
            # wrap plain strings in text Parts.
            raw_parts = m.get("parts")
            if raw_parts and any(
                hasattr(p, "function_call") or hasattr(p, "function_response")
                for p in raw_parts
            ):
                parts = list(raw_parts)
            else:
                content = m.get("content", "") or ""
                if isinstance(content, str):
                    parts = [genai.protos.Part(text=content)] if content else []
                elif isinstance(content, list):
                    parts = [
                        genai.protos.Part(text=b.get("text", ""))
                        for b in content if isinstance(b, dict) and b.get("text")
                    ]
                else:
                    parts = [genai.protos.Part(text=str(content))]

            if not parts:
                continue

            # Merge consecutive same-role messages into one
            if genai_contents and genai_contents[-1]["role"] == gemini_role:
                genai_contents[-1]["parts"].extend(parts)
            else:
                genai_contents.append({"role": gemini_role, "parts": parts})

        sys_instruction = "\n\n".join(sys_parts) if sys_parts else None
        return genai_contents, sys_instruction

    @staticmethod
    def _fc_args_to_dict(args) -> dict:
        """Convert protobuf function-call args to a plain Python dict.

        ``fc.args`` is a proto-plus ``MapComposite`` wrapping a protobuf
        ``Struct``.  A shallow ``dict()`` leaves nested lists as
        ``RepeatedComposite`` objects that are not JSON-serializable.
        ``MessageToDict`` handles the full recursive conversion.
        """
        if not args:
            return {}
        try:
            from google.protobuf.json_format import MessageToDict
            pb = args._pb if hasattr(args, "_pb") else args
            return MessageToDict(pb)
        except Exception:
            return {str(k): v for k, v in args.items()}

    def complete(self, messages, system=None, tools=None) -> LLMResponse:
        genai = self._genai
        contents, sys_instruction = self._to_gemini_messages(messages, system)
        converted_tools = self._convert_tools(tools)

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=sys_instruction,
        )

        kwargs: dict[str, Any] = {
            "generation_config": genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0.2,
            ),
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        response = model.generate_content(contents, **kwargs)

        content_text = ""
        tc_list = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    fc = part.function_call
                    args_dict = self._fc_args_to_dict(fc.args)
                    tc_list.append(ToolCall(
                        id=f"gemini_{uuid.uuid4().hex[:12]}",
                        name=fc.name,
                        arguments=json.dumps(args_dict),
                    ))
                elif hasattr(part, "text") and part.text:
                    content_text += part.text

        stop = "tool_calls" if tc_list else "stop"
        return LLMResponse(
            content=content_text or None,
            tool_calls=tc_list,
            stop_reason=stop,
            raw=response,
        )

    def complete_structured(self, messages, response_model, system=None):
        genai = self._genai
        contents, sys_instruction = self._to_gemini_messages(messages, system)

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=sys_instruction,
        )

        schema = response_model.model_json_schema()

        def _clean_schema(d):
            """Strip JSON Schema keywords unsupported by Gemini.

            Only removes ``default`` and ``title`` from *property definition*
            dicts (those containing ``type``), not from ``properties``
            containers where they may be legitimate property names.
            """
            if isinstance(d, dict):
                d.pop("additionalProperties", None)
                if "type" in d:
                    d.pop("default", None)
                    d.pop("title", None)
                for v in d.values():
                    _clean_schema(v)
                if "required" in d and "properties" in d:
                    d["required"] = [
                        r for r in d["required"] if r in d["properties"]
                    ]
                    if not d["required"]:
                        del d["required"]
            elif isinstance(d, list):
                for v in d:
                    _clean_schema(v)
        _clean_schema(schema)

        schema_has_refs = isinstance(schema, dict) and (
            ("$defs" in schema) or ("$ref" in json.dumps(schema))
        )

        if schema_has_refs:
            json_schema_hint = json.dumps(schema, indent=2)
            last_content = contents[-1]["parts"][0] if contents else ""
            prompt_to_send = (
                f"{last_content}\n\n"
                "Return ONLY valid JSON matching this JSON Schema (no markdown, no extra text):\n"
                f"{json_schema_hint}"
            )
            response = model.generate_content(
                prompt_to_send,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0.5,
                    response_mime_type="application/json",
                ),
            )
        else:
            response = model.generate_content(
                contents,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0.5,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )

        raw_text = getattr(response, "text", None) or ""
        return _parse_structured(raw_text, response_model)

    def assistant_message(self, response: LLMResponse) -> dict:
        genai = self._genai
        # Preserve the raw API response content (including thought_signature
        # metadata on functionCall parts required by Gemini 3 thinking models).
        raw = response.raw
        if raw and hasattr(raw, "candidates") and raw.candidates:
            raw_content = raw.candidates[0].content
            return {"role": raw_content.role or "model",
                    "parts": list(raw_content.parts)}

        # Fallback for synthetic / non-Gemini responses
        parts = []
        if response.content:
            parts.append(genai.protos.Part(text=response.content))
        for tc in (response.tool_calls or []):
            args = json.loads(tc.arguments) if tc.arguments else {}
            parts.append(genai.protos.Part(
                function_call=genai.protos.FunctionCall(
                    name=tc.name, args=args,
                )
            ))
        if not parts:
            parts.append(genai.protos.Part(text=""))
        return {"role": "model", "parts": parts}

    def tool_result_message(self, tool_call: ToolCall, result: str) -> dict:
        genai = self._genai
        return {
            "role": "user",
            "parts": [genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=tool_call.name,
                    response={"result": result},
                )
            )],
        }

    def complete_vision(self, text, image_bytes, system=None) -> LLMResponse:
        genai = self._genai
        image_part = genai.protos.Part(
            inline_data={"mime_type": "image/png", "data": image_bytes}
        )
        text_part = genai.protos.Part(text=text)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
        )
        response = model.generate_content(
            [{"role": "user", "parts": [text_part, image_part]}],
            generation_config=genai.types.GenerationConfig(
                candidate_count=1, temperature=0.3,
            ),
        )
        content_text = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content_text += part.text
        return LLMResponse(
            content=content_text or None,
            stop_reason="stop",
            raw=response,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(model: str) -> LLMClient:
    """Create an LLMClient for the given model name.

    Dispatches by model prefix (any suffix is accepted, including dated snapshots):
        gpt-*, o1*, o3*   -> OpenAI
        claude*            -> Anthropic (e.g. ``claude-opus-4-6``, ``claude-sonnet-4-6``,
                            ``claude-sonnet-4-20250514``)
        gemini*            -> Google Gemini (e.g. ``gemini-3.1-pro-preview``,
                            ``gemini-2.5-pro``)
    """
    if model.startswith(("gpt-", "o1", "o3")) or "/" in model:
        return OpenAIClient(model)
    elif model.startswith("claude"):
        return AnthropicClient(model)
    elif model.startswith("gemini"):
        return GeminiClient(model)
    raise ValueError(
        f"Unknown model prefix: {model!r}. "
        "Expected gpt-*, o1*, o3*, claude*, or gemini* (any version, e.g. gemini-3.1-pro-preview)."
    )
