"""Agent loop tests with a scripted fake LLM (no network, no API key)."""

from emporio_agent.agent import FALLBACK_ANSWER, Agent
from emporio_agent.llm import LLMResponse, ToolCall
from emporio_agent.memory import ConversationStore


class ScriptedLLM:
    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system, messages, tools):
        self.calls.append({"system": system, "messages": list(messages), "tools": [t.name for t in tools]})
        return self.responses.pop(0)


def make_agent(llm, tools):
    return Agent(llm, tools=tools, memory=ConversationStore(":memory:"))


def test_tool_call_then_answer(tools):
    llm = ScriptedLLM([
        LLMResponse(None, [ToolCall("1", "get_product_details", {"product": "Takamine GD20"})]),
        LLMResponse("O Takamine GD20 sai por R$ 2.199,00."),
    ])
    reply = make_agent(llm, tools).reply("s1", "Quanto custa o Takamine GD20?")
    assert reply.text == "O Takamine GD20 sai por R$ 2.199,00."
    assert [t.name for t in reply.trace] == ["get_product_details"]
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"]["produtos"][0]["preco_final"] == "R$ 2.199,00"


def test_system_prompt_contains_live_store_status(tools):
    llm = ScriptedLLM([LLMResponse("Oi!")])
    make_agent(llm, tools).reply("s1", "oi")
    assert "terça-feira, 24/03/2026" in llm.calls[0]["system"] and "ABERTA" in llm.calls[0]["system"]


def test_history_is_persisted_without_tool_payloads(tools):
    llm = ScriptedLLM([
        LLMResponse(None, [ToolCall("1", "search_products", {"category": "Ukuleles"})]),
        LLMResponse("Temos vários ukuleles!"),
        LLMResponse("O mais barato é o Shelby."),
    ])
    agent = make_agent(llm, tools)
    agent.reply("s1", "tem ukulele?")
    agent.reply("s1", "qual o mais barato?")
    second_turn = llm.calls[2]["messages"]
    assert [m["role"] for m in second_turn] == ["user", "assistant", "user"]
    assert second_turn[1]["content"] == "Temos vários ukuleles!"


def test_runaway_tool_loop_falls_back_gracefully(tools):
    looping = [LLMResponse(None, [ToolCall(str(i), "list_active_promotions", {})]) for i in range(10)]
    agent = Agent(ScriptedLLM(looping), tools=tools, memory=ConversationStore(":memory:"), max_steps=3)
    assert agent.reply("s1", "promo?").text == FALLBACK_ANSWER


def test_llm_error_falls_back_gracefully(tools):
    class Broken:
        name = "broken"

        def generate(self, *a):
            raise RuntimeError("boom")

    assert make_agent(Broken(), tools).reply("s1", "oi").text == FALLBACK_ANSWER
