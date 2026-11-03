"""Provider-agnostic LLM interface with two adapters.

The agent loop speaks a tiny neutral message format; each adapter converts it to its provider's
SDK. This keeps the agent testable with a fake LLM and lets the store swap models (free Gemini tier
for the prototype, OpenAI/Groq/local Ollama later) by changing an environment variable.

Neutral messages:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str | None, "tool_calls": [ToolCall], "raw": <provider object>}
    {"role": "tool", "tool_call_id": str, "name": str, "content": dict}
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import Tool


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider-native assistant message, replayed verbatim inside the same turn


class LLM(Protocol):
    name: str

    def generate(self, system: str, messages: list[dict], tools: list[Tool]) -> LLMResponse: ...


def _retry(fn, attempts: int = 4):
    """Free tiers rate-limit aggressively; back off on 429/5xx instead of failing the conversation."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # SDK-specific exception types differ; inspect the message
            msg = str(exc)
            transient = any(code in msg for code in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"))
            if not transient or i == attempts - 1:
                raise
            time.sleep(min(2 ** i * 5, 40))


class GeminiLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.3):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.name = f"gemini:{self.model}"
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.temperature = temperature

    def _contents(self, messages: list[dict]) -> list:
        t = self._types
        contents = []
        for m in messages:
            if m["role"] == "user":
                contents.append(t.Content(role="user", parts=[t.Part.from_text(text=m["content"])]))
            elif m["role"] == "assistant":
                if m.get("raw") is not None:  # keeps Gemini "thought signatures" intact
                    contents.append(m["raw"])
                else:
                    contents.append(t.Content(role="model", parts=[t.Part.from_text(text=m["content"] or "")]))
            elif m["role"] == "tool":
                part = t.Part.from_function_response(name=m["name"], response={"result": m["content"]})
                if contents and contents[-1].role == "user" and contents[-1].parts[0].function_response:
                    contents[-1].parts.append(part)  # parallel calls -> one turn with all responses
                else:
                    contents.append(t.Content(role="user", parts=[part]))
        return contents

    def generate(self, system: str, messages: list[dict], tools: list[Tool]) -> LLMResponse:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=tool.name, description=tool.description, parameters_json_schema=tool.parameters)
                for tool in tools
            ])],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = _retry(lambda: self.client.models.generate_content(model=self.model, contents=self._contents(messages), config=config))
        candidate = resp.candidates[0] if resp.candidates else None
        content = candidate.content if candidate else None
        calls, texts = [], []
        for part in (content.parts if content and content.parts else []):
            if part.function_call:
                fc = part.function_call
                calls.append(ToolCall(id=fc.id or uuid.uuid4().hex[:8], name=fc.name, arguments=dict(fc.args or {})))
            elif part.text and not part.thought:
                texts.append(part.text)
        return LLMResponse(text="".join(texts).strip() or None, tool_calls=calls, raw=content)


class OpenAICompatibleLLM:
    """Works with OpenAI, Groq, OpenRouter, Together, a local Ollama server (`/v1`), etc."""

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None, temperature: float = 0.3):
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.name = f"openai-compatible:{self.model}"
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed", base_url=base_url or os.getenv("OPENAI_BASE_URL") or None)
        self.temperature = temperature

    def generate(self, system: str, messages: list[dict], tools: list[Tool]) -> LLMResponse:
        wire: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                        for c in m["tool_calls"]
                    ]
                wire.append(msg)
            elif m["role"] == "tool":
                wire.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": json.dumps(m["content"], ensure_ascii=False)})

        resp = _retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=wire,
            temperature=self.temperature,
            tools=[{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools],
        ))
        message = resp.choices[0].message
        calls = []
        for c in message.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args))
        return LLMResponse(text=(message.content or "").strip() or None, tool_calls=calls)


def make_llm(provider: str | None = None) -> LLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider == "gemini":
        return GeminiLLM()
    if provider in ("openai", "openai-compatible", "ollama", "groq"):
        return OpenAICompatibleLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
