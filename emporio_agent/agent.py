"""The agent loop: a plain function-calling (ReAct-style) loop, no framework.

    user message -> LLM decides -> [tool calls -> tool results -> LLM decides]* -> final answer

Why no LangChain/LangGraph: the whole loop is ~40 lines, fully visible, easy to debug and to test
with a fake LLM, and has no hidden prompts. A framework would add value with many agents/graphs,
not with one agent and seven tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .llm import LLM
from .memory import ConversationStore
from .prompts import build_system_prompt
from .tools import StoreTools

log = logging.getLogger(__name__)

FALLBACK_ANSWER = (
    "Desculpa, tive um probleminha aqui pra concluir sua solicitação 😕 Pode tentar de novo? "
    "Se preferir, fala com a equipe pelo (67) 3341-4444."
)


@dataclass
class ToolTrace:
    name: str
    arguments: dict
    result: dict


@dataclass
class AgentReply:
    text: str
    trace: list[ToolTrace] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLM, tools: StoreTools | None = None, memory: ConversationStore | None = None, max_steps: int = 6):
        self.llm = llm
        self.tools = tools or StoreTools()
        self.memory = memory or ConversationStore()
        self.max_steps = max_steps

    def reply(self, session_id: str, user_text: str) -> AgentReply:
        messages = self.memory.history(session_id) + [{"role": "user", "content": user_text}]
        specs = self.tools.specs()
        system = build_system_prompt()
        trace: list[ToolTrace] = []
        answer = None

        for _ in range(self.max_steps):
            try:
                response = self.llm.generate(system, messages, specs)
            except Exception:
                log.exception("LLM call failed")
                break
            if not response.tool_calls:
                answer = response.text
                break
            messages.append({"role": "assistant", "content": response.text, "tool_calls": response.tool_calls, "raw": response.raw})
            for call in response.tool_calls:
                result = self.tools.call(call.name, call.arguments)
                trace.append(ToolTrace(call.name, call.arguments, result))
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})

        if not answer:
            answer = FALLBACK_ANSWER
        self.memory.append(session_id, "user", user_text)
        self.memory.append(session_id, "assistant", answer)
        return AgentReply(answer, trace)
