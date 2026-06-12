import os
from typing_extensions import TypedDict
from datetime import datetime

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient
from openai import OpenAI

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# state

class AgentState(TypedDict):
    query: str
    route: str
    answer: str
    context: str

# nodes
def decide(state):
    query = state["query"].lower()

    if "weather" in query:
        return {"route": "weather"}
    
    if "time" in query:
        return {"route": "time"}
    
    return {"route": "llm"}

def route_query(state: AgentState):
    return state["route"]

def weather_node(state: AgentState):
    search_query = state["query"]

    result = tavily.search(
        query=search_query,
        max_results=1
    )

    results = result.get("results", [])

    if not results:
        return {
            "answer": "I could not find weather information"
        }

    weather_summary = "\n\n".join(
        item.get("content", "") for item in results
    )

    return {
    "context": weather_summary
    }
   
def time_node(state: AgentState):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "answer": f"The current system time is {current_time}"
    }    

def llm_node(state: AgentState):
    if state.get("context"):
        prompt = f"""
User question:
{state["query"]}

Context:
{state["context"]}

Give a simple human-friendly answer.
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

builder.add_node("decide", decide)
builder.add_node("weather", weather_node)
builder.add_node("time", time_node)
builder.add_node("llm", llm_node)

builder.add_edge(START, "decide")
builder.add_conditional_edges(
    "decide",
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
with open("manual_router_graph.png", "wb") as f:
    f.write(png_data)
print("Graph image saved as manual_router_graph.png")

if __name__ == "__main__":
    result = graph.invoke({
        "query": "What is the weather in chennai?",
        "route": "",
        "answer": "",
    })
    print("\nFinal Answer:")
    print(result["answer"])