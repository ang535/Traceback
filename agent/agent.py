import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from agent.tools import TOOLS

load_dotenv()


def build_agent():
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    agent = create_react_agent(llm, TOOLS)
    return agent


def run_agent(task: str) -> dict:
    agent = build_agent()
    initial_state = {"messages": [{"role": "user", "content": task}]}
    final_state = agent.invoke(initial_state)
    return final_state