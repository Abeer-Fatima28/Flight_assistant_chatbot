from dotenv import load_dotenv
load_dotenv()

from logger_config import setup_logger
logger = setup_logger()

from graph.langgraph_app import build_graph
from memory.memory import ConversationMemory


def chat(app):
    logger.info("Starting chat loop.")
    print("LLM Agents (Multi-Prompt ReAct + Sliding Memory, GPT-4o-mini) ready. Type 'exit' to quit.\n")
    mem = ConversationMemory()
    state = {'messages': [], 'memory': mem}
    while True:
        try:
            q = input("You: ")
        except EOFError:
            break
        if q.strip().lower() == 'exit':
            break
        logger.info(f"User query: {q}")
        mem.add_user(q)
        state['query'] = q
        out = app.invoke(state)
        resp = out.get('response') or "(No textual summary produced — but the agent returned structured results.)"
        print("Bot:", resp)
        logger.info(f"Response: {resp}")
        state = {**out, 'memory': mem}


if __name__ == '__main__':
    app = build_graph()
    chat(app)
