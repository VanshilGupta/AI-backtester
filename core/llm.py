"""Provider-agnostic LLM access.

Two operations the app needs:
  * `generate_json`  - a structured strategy spec conforming to a JSON schema.
  * `generate_text`  - a free-text qualitative assessment.

Two providers, selectable by the user:
  * Anthropic (Claude) - official `anthropic` SDK. Uses adaptive thinking,
    prompt caching on the (stable) system block, and the structured
    `output_config` json_schema path so we get a typed object back.
  * Google (Gemini)    - official `google-genai` SDK. Uses JSON response mode
    with an explicit key envelope derived from the same schema.

Both raise a unified `LLMError` so callers don't depend on provider internals.
SDKs are imported lazily so a missing optional package only matters when that
provider is actually selected.
"""
from __future__ import annotations

import json
import re

ANTHROPIC = "Anthropic (Claude)"
GOOGLE = "Google (Gemini)"

PROVIDER_MODELS: dict[str, list[str]] = {
    ANTHROPIC: ["claude-opus-4-7", "claude-sonnet-4-6"],
    GOOGLE: ["gemini-2.5-pro", "gemini-2.5-flash"],
}

# (provider -> env vars checked, first non-empty wins) for prefilling the UI.
PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    ANTHROPIC: ["ANTHROPIC_API_KEY"],
    GOOGLE: ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}

DEFAULT_PROVIDER = ANTHROPIC


class LLMError(Exception):
    """Generation failed (auth, transport, refusal, or unparseable output)."""


# --- shared helpers ----------------------------------------------------------
def _json_envelope_hint(schema: dict) -> str:
    """Human-readable 'return exactly these keys' note built from the schema,
    so the Gemini path stays in lockstep with the Anthropic json_schema."""
    props = schema.get("properties", {})
    required = schema.get("required", list(props))
    lines = []
    for key in required:
        spec = props.get(key, {})
        typ = spec.get("type", "string")
        if "enum" in spec:
            typ = "one of: " + ", ".join(spec["enum"])
        elif typ == "array":
            typ = f"array of {spec.get('items', {}).get('type', 'string')}"
        lines.append(f'  "{key}": <{typ}>')
    return (
        "Return ONLY a single JSON object (no markdown, no code fences, no "
        "commentary) with exactly these keys:\n{\n"
        + ",\n".join(lines)
        + "\n}"
    )


def _loads_lenient(text: str) -> dict:
    """Parse JSON that may be wrapped in prose or ```json fences."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            return json.loads(t[start : end + 1])
        raise


# --- Anthropic ---------------------------------------------------------------
def _anthropic_client(api_key: str | None):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - env dependent
        raise LLMError(
            "The 'anthropic' package is required. Install with: "
            "python -m pip install anthropic"
        ) from exc
    return anthropic, (
        anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    )


def _anthropic_json(model, system, user, schema, api_key) -> dict:
    anthropic, client = _anthropic_client(api_key)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={
                "effort": "high",
                "format": {"type": "json_schema", "schema": schema},
            },
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError as exc:
        raise LLMError("Invalid or missing Anthropic API key.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(
            f"Anthropic API error ({exc.status_code}): {exc.message}"
        ) from exc

    if message.stop_reason == "refusal":
        raise LLMError("The model declined to generate this strategy.")
    for block in message.content:
        if block.type == "text":
            return _loads_lenient(block.text)
    raise LLMError("Anthropic returned no text block to parse.")


def _anthropic_text(model, system, user, max_tokens, api_key) -> str:
    anthropic, client = _anthropic_client(api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user}],
    )
    return next(
        (b.text for b in message.content if b.type == "text"), ""
    ).strip()


# --- Google Gemini -----------------------------------------------------------
def _gemini():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMError(
            "Google Gemini support needs the 'google-genai' package. Install "
            "with: python -m pip install google-genai"
        ) from exc
    return genai, types


def _gemini_generate(model, system, user, api_key, json_mode: bool) -> str:
    genai, types = _gemini()
    if not api_key:
        raise LLMError("A Google Gemini API key is required.")
    try:
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = client.models.generate_content(
            model=model, contents=user, config=config
        )
    except Exception as exc:  # google.genai.errors.* + transport
        msg = str(exc)
        if "API_KEY" in msg.upper() or "401" in msg or "403" in msg:
            raise LLMError("Invalid or missing Google Gemini API key.") from exc
        raise LLMError(f"Gemini API error: {msg}") from exc

    text = (resp.text or "").strip()
    if not text:
        raise LLMError("Gemini returned an empty response.")
    return text


# --- Public dispatch ---------------------------------------------------------
def generate_json(
    provider: str,
    model: str,
    system: str,
    user: str,
    schema: dict,
    api_key: str | None = None,
) -> dict:
    if provider == ANTHROPIC:
        return _anthropic_json(model, system, user, schema, api_key)
    if provider == GOOGLE:
        sys_with_envelope = system + "\n\n" + _json_envelope_hint(schema)
        raw = _gemini_generate(model, sys_with_envelope, user, api_key, True)
        try:
            return _loads_lenient(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Gemini did not return valid JSON: {exc}") from exc
    raise LLMError(f"Unknown provider: {provider}")


def generate_text(
    provider: str,
    model: str,
    system: str,
    user: str,
    api_key: str | None = None,
    max_tokens: int = 1024,
) -> str:
    if provider == ANTHROPIC:
        return _anthropic_text(model, system, user, max_tokens, api_key)
    if provider == GOOGLE:
        return _gemini_generate(model, system, user, api_key, False)
    raise LLMError(f"Unknown provider: {provider}")
