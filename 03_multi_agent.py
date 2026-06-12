import os
import json
from typing_extensions import TypedDict
from datetime import datetime

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient
from openai import OpenAI

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AgentState(TypedDict):
    query: str
    route: str
    context: str
    answer: str


def llm_router(state: AgentState):
    prompt = f"""
You are a router.

Choose one route for the user query.

Available routes:
- weather: for weather, rain, temperature, climate today
- time: for current time
- llm: for general questions

Return only JSON in this format:
{{"route": "weather"}}

User query:
{state["query"]}
"""

    response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    raw_output = response.output_text.strip()
    parsed = json.loads(raw_output)

    return {
        "route": parsed["route"]
    }


def route_query(state: AgentState):
    return state["route"]


def weather_node(state: AgentState):
    result = tavily.search(
        query=state["query"],
        max_results=3
    )

    results = result.get("results", [])

    if not results:
        return {
            "context": "No weather information found."
        }

    weather_context = "\n\n".join(
        item.get("content", "") for item in results
    )

    print(weather_context)

    return {
        "context": weather_context
    }


def time_node(state: AgentState):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "answer": f"The current system time is {current_time}"
    }


def llm_node(state: AgentState):
    context = state.get("context", "")

    if context:
        prompt = f"""
User question:
{state["query"]}

Context from tool:
{context}

Give a simple, human-friendly answer.
"""
    else:
        prompt = state["query"]

    response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        "answer": response.output_text
    }


builder = StateGraph(AgentState)

builder.add_node("llm_router", llm_router)
builder.add_node("weather", weather_node)
builder.add_node("time", time_node)
builder.add_node("llm", llm_node)

builder.add_edge(START, "llm_router")

builder.add_conditional_edges(
    "llm_router",
    route_query,
    {
        "weather": "weather",
        "time": "time",
        "llm": "llm",
    },
)

builder.add_edge("weather", "llm")
builder.add_edge("time", END)
builder.add_edge("llm", END)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

png_data = graph.get_graph().draw_mermaid_png()
with open("llm_router_graph.png", "wb") as f:
    f.write(png_data)

print("Graph image saved as llm_router_graph.png")


if __name__ == "__main__":
    result = graph.invoke({
        "query": "Is it raining in Chennai today?",
        "route": "",
        "context": "",
        "answer": "",
    })

    print("\nRoute:")
    print(result["route"])

    print("\nFinal Answer:")
    print(result["answer"])