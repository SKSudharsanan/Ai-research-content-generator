import os
from datetime import datetime
from typing import Annotated, Literal

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

import gradio as gr
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_current_time() -> str:
    """Use this tool when the user asks for current date or time."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current system date and time is {current_time}"


@tool
def calculator(expression: str) -> str:
    """Use this tool for math calculations."""
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
    """Use this tool when the user asks to save, write, or create a local file."""
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
    streaming=True,
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

    return {"messages": [response]}


def human_approval_node(state: AgentState):
    last_message = state["messages"][-1]

    tool_calls = getattr(last_message, "tool_calls", [])

    approval = interrupt(
        {
            "question": "Do you approve running these tool calls?",
            "tool_calls": tool_calls,
        }
    )

    if approval is True:
        return {}

    return {
        "messages": [
            AIMessage(content="Tool call cancelled by human approval.")
        ]
    }


def should_continue(state: AgentState) -> Literal["human_approval", "end"]:
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "human_approval"

    return "end"


def after_approval(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"


builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "human_approval": "human_approval",
        "end": END,
    },
)

builder.add_conditional_edges(
    "human_approval",
    after_approval,
    {
        "tools": "tools",
        "end": END,
    },
)

builder.add_edge("tools", "agent")


memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory,
)


def run_graph_with_streaming(user_input: str, thread_id: str = "cli-user"):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    graph_input = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }

    final_answer = ""

    while True:
        interrupted = False

        for event in graph.stream(
            graph_input,
            config=config,
            stream_mode="updates",
        ):
            if "__interrupt__" in event:
                interrupted = True

                interrupt_data = event["__interrupt__"][0].value

                print("\nHuman approval required:")
                print(interrupt_data["question"])

                for tool_call in interrupt_data["tool_calls"]:
                    print(f"\nTool: {tool_call['name']}")
                    print(f"Args: {tool_call['args']}")

                approval_input = input("\nApprove tool call? yes/no: ").strip().lower()
                approved = approval_input in ["yes", "y"]

                graph_input = Command(resume=approved)
                break

            for node_name, node_output in event.items():
                if "messages" in node_output:
                    last_message = node_output["messages"][-1]

                    if last_message.content:
                        final_answer = last_message.content
                        print(f"\n[{node_name}]")
                        print(last_message.content)

        if not interrupted:
            break

    return final_answer


def cli_loop():
    print("\nLangGraph Agent CLI")
    print("Type 'exit' to quit.\n")

    thread_id = input("Enter thread id, default cli-user: ").strip() or "cli-user"

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        run_graph_with_streaming(user_input, thread_id=thread_id)


def launch_gradio():
    def chat_fn(message, history=None):
        thread_id = "gradio-user"

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        graph_input = {
            "messages": [
                HumanMessage(content=message)
            ]
        }

        response_text = ""

        while True:
            interrupted = False

            for event in graph.stream(
                graph_input,
                config=config,
                stream_mode="updates",
            ):
                print("\nEVENT:", event)

                # Handle interrupts
                if "__interrupt__" in event:
                    interrupted = True

                    interrupt_data = event["__interrupt__"][0].value

                    response_text += "\n\n🔧 Tool approval required\n"

                    for tool_call in interrupt_data.get("tool_calls", []):
                        response_text += (
                            f"\nTool: {tool_call['name']}"
                            f"\nArgs: {tool_call['args']}\n"
                        )

                    yield response_text

                    # Auto approve in Gradio
                    graph_input = Command(resume=True)
                    break

                # Normal node updates
                for node_name, node_output in event.items():

                    if node_output is None:
                        continue

                    if not isinstance(node_output, dict):
                        continue

                    messages = node_output.get("messages")

                    if not messages:
                        continue

                    last_message = messages[-1]

                    if hasattr(last_message, "content") and last_message.content:

                        if isinstance(last_message.content, str):
                            response_text = last_message.content
                            yield response_text

            if not interrupted:
                break

        yield response_text

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="LangGraph Agent",
        description="LangGraph agent with memory, streaming, tools, and human approval.",
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

if __name__ == "__main__":
    mode = input("Choose mode: cli / gradio: ").strip().lower()

    if mode == "gradio":
        launch_gradio()
    else:
        cli_loop()