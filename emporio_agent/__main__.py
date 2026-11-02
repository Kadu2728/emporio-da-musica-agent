"""Interactive CLI:  python -m emporio_agent [--session ID] [--debug] [--provider gemini|openai]"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Tom, o assistente virtual da Empório da Música")
    parser.add_argument("--session", help="Session id to resume a conversation (e.g. the customer's phone).")
    parser.add_argument("--provider", help="LLM provider: gemini (default) or openai.")
    parser.add_argument("--debug", action="store_true", help="Show tool calls and their arguments.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")

    from .agent import Agent
    from .llm import make_llm

    agent = Agent(make_llm(args.provider))
    session = args.session or uuid.uuid4().hex[:8]
    print(f"Empório da Música — assistente virtual Tom  (modelo: {agent.llm.name}, sessão: {session})")
    print("Digite sua mensagem. Comandos: /novo (nova conversa), /sair\n")

    while True:
        try:
            text = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/sair", "/exit", "/quit"):
            break
        if text == "/novo":
            agent.memory.clear(session)
            print("(conversa reiniciada)\n")
            continue
        reply = agent.reply(session, text)
        if args.debug:
            for step in reply.trace:
                print(f"  [tool] {step.name}({json.dumps(step.arguments, ensure_ascii=False)})")
        print(f"\nTom: {reply.text}\n")


if __name__ == "__main__":
    main()
