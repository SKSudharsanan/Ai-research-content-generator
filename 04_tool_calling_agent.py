import os
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_current_time() -> str:
    """
    Use this tool when the user asks for the current date or current time.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current system date and time is {current_time}"


@tool
def calculator(expression: str) -> str:
    """
    Use this tool for math calculations.

    Input should be a valid Python math expression.
    Example:
    10 + 20
    5000 * 12
    (15000 * 18) / 100
    """
    try:
        allowed_chars = "0123456789+-*/(). %"
        if not all(char in allowed_chars for char in expression):
            return "Invalid expression. Only numbers and basic math operators are allowed."

        result = eval(expression)

        return f"The result is {result}"

    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def write_file(filename: str, content: str) -> str:
    """
    Use this tool when the user asks to save, write, or create a local file.

    Inputs:
    - filename: name of the file to create
    - content: content to write into the file
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)

        return f"File written successfully: {filename}"

    except Exception as e:
        return f"File writing error: {str(e)}"


tavily_search = TavilySearch(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)


tools = [
    tavily_search,
    get_current_time,
    calculator,
    write_file,
]


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)


llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    system_prompt = SystemMessage(
        content="""
You are a helpful AI assistant.

You have access to tools.

Use tavily_search_results_json when the user asks about:
- weather
- latest news
- current events
- live information
- anything that may require web search

Use get_current_time when the user asks about:
- current time
- today's date
- current date

Use calculator when the user asks for:
- math
- arithmetic
- percentages
- financial calculations
- numeric calculations

Use write_file when the user asks to:
- save something
- create a file
- write content into a file

If no tool is needed, answer directly.
"""
    )

    messages = [system_prompt] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"


builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

builder.add_edge("tools", "agent")

graph = builder.compile()


print(graph.get_graph().draw_mermaid())

png_data = graph.get_graph().draw_mermaid_png()

with open("tool_calling_4_tools_graph.png", "wb") as f:
    f.write(png_data)

print("Graph image saved as tool_calling_4_tools_graph.png")


if __name__ == "__main__":
    user_query = """
What is the weather in Chennai today?
Also tell me the current time.
Calculate 15000 * 12.
Then save the final answer into output.md.
"""

    result = graph.invoke({
        "messages": [
            HumanMessage(content=user_query)
        ]
    })

    print("\nFinal Answer:")
    print(result["messages"][-1].content)