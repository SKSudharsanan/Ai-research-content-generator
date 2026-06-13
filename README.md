# Agentic AI System with LangGraph

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Framework-green)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM-black)
![Tavily](https://img.shields.io/badge/Tavily-Search-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

Workshop code from AI Dev Day organized by the Chennai Robotics and AI Builders Community.

## Agentic AI System with LangGraph

This repository contains the code examples used for my AI Dev Day session on building Agentic AI systems with LangGraph.

The goal of this repo is to help beginners and builders understand how AI agents are built step by step, starting from structured outputs, moving into graph-based workflows, and finally building tool-calling agents with memory, streaming, and human approval.

## Talk Recordings

### YouTube Part 1
https://www.youtube.com/live/hoC4SZJeYL8

### YouTube Part 2
https://www.youtube.com/live/ICsARkO-6iY

### Facebook Part 1
https://www.facebook.com/reel/1028171163086388

### Facebook Part 2
https://www.facebook.com/reel/1301304925037374

## What This Repo Covers

* Pydantic basics for structured data
* LLM structured output
* LangGraph state management
* Manual routing with graph nodes
* LLM-based routing
* Multi-agent style workflows
* Tool calling agents
* Tavily web search integration
* Current time tool
* Calculator tool
* File writing tool
* Human approval before tool execution
* Memory using LangGraph checkpointer
* Streaming responses
* CLI and Gradio interface

## Files

### 01_pydantic_basics.py

Introduces Pydantic models and validation.

It shows how to:

* Create a structured schema using BaseModel
* Validate fields using Field
* Add custom validation using field_validator
* Catch validation errors
* Use OpenAI structured output with Pydantic

### 02_simple_flow.py

A simple LangGraph workflow with manual routing.

It includes:

* State definition using TypedDict
* A decision node
* Weather route
* Time route
* LLM fallback route
* Graph compilation
* Mermaid graph generation

Generated graph image:

manual_router_graph.png

### 03_multi_agent.py

An upgraded version where the routing decision is made by the LLM.

It includes:

* LLM-based router
* JSON route selection
* Weather search using Tavily
* Time node
* LLM response generation
* Conditional graph edges

Generated graph image:

llm_router_graph.png

### 04_tool_calling_agent.py

A tool-calling LangGraph agent.

The agent has access to:

* Tavily Search
* Current time tool
* Calculator tool
* File writer tool

It demonstrates how an agent can decide when to call tools and loop back to the LLM after tool execution.

Generated graph image:

tool_calling_4_tools_graph.png

### demo.py

The final demo version.

It includes:

* Tool calling
* Memory with InMemorySaver
* Streaming graph execution
* Human approval before tool execution
* CLI mode
* Gradio UI mode

This is the most complete example in the repo.

### slides.html

HTML slide deck used for presenting the workshop.

## Setup

### Clone the repository:

```bash
git clone https://github.com/SKSudharsanan/Agentic-AI-System-with-Langgraph.git
cd Agentic-AI-System-with-Langgraph
```

### Create a virtual environment:

```bash
python -m venv venv
```

### Activate the virtual environment:

#### For macOS/Linux:

```bash
source venv/bin/activate
```

#### For Windows:

```bash
venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Create a .env file:

```bash
cp .env.example .env
```

### How to Run

#### Run the Pydantic example:

```bash
python 01_pydantic_basics.py
```

#### Run the simple manual router:

```bash
python 02_simple_flow.py
```

#### Run the LLM router example:

```bash
python 03_multi_agent.py
```

#### Run the tool-calling agent:

```bash
python 04_tool_calling_agent.py
```

#### Run the final demo:

```bash
python demo.py
```

Then choose:

cli

or

gradio

If you choose Gradio, open the local URL shown in the terminal.

Example Prompt for demo.py

What is the weather in Chennai today?
Also tell me the current time.
Calculate 15000 * 12.
Then save the final answer into output.md.

The agent will decide which tools to use, ask for approval, execute the tools, and return the final answer.

## Core Idea

A normal chatbot only responds to a message.

An agentic AI system can:

* Understand the user request
* Decide what path to take
* Call tools when needed
* Use external information
* Remember state
* Ask for human approval
* Continue execution after tool results
* Produce a final useful output

LangGraph helps us build these systems as graphs, where each node has a clear responsibility.

## Requirements

* Python 3.10+
* OpenAI API key
* Tavily API key
* Basic understanding of Python
* Basic understanding of LLMs

## Connect With Me

### YouTube
https://www.youtube.com/channel/UCJI3CThQtWbDLohsiMs1ryA

### Instagram
https://www.instagram.com/sudharsanan_kirubanandhan/

### Facebook
https://www.facebook.com/sudharsanankirubanandhan/

### Linkedin
https://linkedin.com/in/sksudharsanan

### X
https://x.com/sksudharsanan

### GitHub
https://github.com/SKSudharsanan

## About

This repository contains the code examples, demos, and workshop materials used during my session at AI Dev Day, organized by the Chennai Robotics and AI Builders Community.

Please do follow me on Youtube, Instagram, Linkedin, X, Github and Facebook

I share AI experiments, agent-building tutorials, engineering lessons, and opinions around practical AI system design.
