from typing import List, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
import playwright
from IPython import display
from playwright.async_api import async_playwright
import base64
import re

from browser_annotations import mark_page
from WebVoyer_actions import click, type_text, scroll, wait, go_back, to_google
from schemas import *

class WebVoyager:
    async def annotate(state):
        marked_page = await mark_page.with_retry().ainvoke(state["page"])
        return {**state, **marked_page}


    def format_descriptions(state):
        labels = []
        for i, bbox in enumerate(state["bboxes"]):
            text = bbox.get("ariaLabel") or ""
            if not text.strip():
                text = bbox["text"]
            el_type = bbox.get("type")
            labels.append(f'{i} (<{el_type}/>): "{text}"')
        bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
        return {**state, "bbox_descriptions": bbox_descriptions}


    def parse(text: str) -> dict:
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
        action_block = text.strip().split("\n")[-1]

        action_str = action_block[len(action_prefix) :]
        split_output = action_str.split(" ", 1)
        if len(split_output) == 1:
            action, action_input = split_output[0], None
        else:
            action, action_input = split_output
        action = action.strip()
        if action_input is not None:
            action_input = [
                inp.strip().strip("[]") for inp in action_input.strip().split(";")
            ]
        return {"action": action, "args": action_input}


    # Will need a later version of langchain to pull
    # this image prompt template
    prompt = hub.pull("wfh/web-voyager")

    # TODO : Change model to gpt-4-o
    llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=4096)
    agent = annotate | RunnablePassthrough.assign(
        prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
    )

    def update_scratchpad(state: AgentState):
        """After a tool is invoked, we want to update
        the scratchpad so the agent is aware of its previous steps"""
        old = state.get("scratchpad")
        if old:
            txt = old[0].content
            last_line = txt.rsplit("\n", 1)[-1]
            step = int(re.match(r"\d+", last_line).group()) + 1
        else:
            txt = "Previous action observations:\n"
            step = 1
        txt += f"\n{step}. {state['observation']}"

        return {**state, "scratchpad": [SystemMessage(content=txt)]}

    graph_builder = StateGraph(AgentState)


    graph_builder.add_node("agent", agent)
    graph_builder.set_entry_point("agent")

    graph_builder.add_node("update_scratchpad", update_scratchpad)
    graph_builder.add_edge("update_scratchpad", "agent")

    tools = {
        "Click": click,
        "Type": type_text,
        "Scroll": scroll,
        "Wait": wait,
        "GoBack": go_back,
        "Google": to_google,
    }


    for node_name, tool in tools.items():
        graph_builder.add_node(
            node_name,
            # The lambda ensures the function's string output is mapped to the "observation"
            # key in the AgentState
            RunnableLambda(tool) | (lambda observation: {"observation": observation}),
        )
        # Always return to the agent (by means of the update-scratchpad node)
        graph_builder.add_edge(node_name, "update_scratchpad")


    def select_tool(state: AgentState):
        # Any time the agent completes, this function
        # is called to route the output to a tool or
        # to the end user.
        action = state["prediction"]["action"]
        if action == "ANSWER":
            return END
        if action == "retry":
            return "agent"
        return action


    graph_builder.add_conditional_edges("agent", select_tool)

    graph = graph_builder.compile()

    browser = async_playwright().start()
    # We will set headless=False so we can watch the agent navigate the web.
    browser = browser.chromium.launch(headless=False, args=None)
    page = browser.new_page()
    _ = page.goto("https://www.google.com")


    async def call_agent(question: str, page, max_steps: int = 150):
        event_stream = graph.astream(
            {
                "page": page,
                "input": question,
                "scratchpad": [],
            },
            {
                "recursion_limit": max_steps,
            },
        )
        final_answer = None
        steps = []
        async for event in event_stream:
            # We'll display an event stream here
            if "agent" not in event:
                continue
            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")
            display.clear_output(wait=False)
            steps.append(f"{len(steps) + 1}. {action}: {action_input}")
            print("\n".join(steps))
            display.display(display.Image(base64.b64decode(event["agent"]["img"])))
            if "ANSWER" in action:
                final_answer = action_input[0]
                break
        return final_answer
