import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agent.tools import TOOLS

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # use "deepseek-reasoner" for the R1 reasoning model


def build_agent():
    """Build and return the LangGraph coding agent, powered by DeepSeek."""
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )
    agent = create_react_agent(llm, TOOLS)
    return agent


def run_agent(task: str) -> dict:
    """Run the coding agent on a task and return the final state.

    Args:
        task: A natural language description of the coding task,
              e.g. "Fix the bug in tasks/buggy_add.py"

    Returns:
        The final LangGraph state dict containing messages and output.
    """
    agent = build_agent()
    initial_state = {"messages": [{"role": "user", "content": task}]}
    final_state = agent.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Quick smoke test — fix a buggy file
    task = "Read the file tasks/buggy_add.py, find the bug, fix it, and run it to confirm it works."
    result = run_agent(task)
    for message in result["messages"]:
        print(f"[{message.type}]: {message.content}\n")
