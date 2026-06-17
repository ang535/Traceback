import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from agent.tools import TOOLS

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash-lite"


def build_agent():
    """Build and return the LangGraph coding agent, powered by Gemini."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
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