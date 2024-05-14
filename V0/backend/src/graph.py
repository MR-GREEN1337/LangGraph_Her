from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from src.utils import load_env, _print_event, handle_tool_error, create_tool_node_with_fallback
import datetime
import shutil


class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

class Bot:
    

    def __init__(self):
        load_env()
    

    class Assistant:
        def __init__(self, runnable: Runnable):
            self.runnable = runnable

        def __call__(self, state: State, config: RunnableConfig):
            while True:
                user_name = config["configurable"]["user_name"]
                #print(user_name)
                state = {**state, "user_name": user_name}
                result = self.runnable.invoke(state)
                # If the LLM happens to return an empty response, we will re-prompt it
                # for an actual response.
                if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
                ):
                    messages = state["messages"] + [("user", "Respond with a real output.")]
                    state = {**state, "messages": messages}
                else:
                    break
            return {"messages": result}

    def kickstart(self):
        llm = ChatGroq(model_name="llama3-70b-8192", temperature=1)

        primary_assistant_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful human being called Samantha to interact with human nature. "
                    " Use the provided tools to for human needs. "
                    " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                    " If a search comes up empty, expand your search before giving up."
                    "\n\nCurrent user's name:\n\n{user_name}\n Remember it well"
                    "\nCurrent time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.datetime.now())

        tools = [
            TavilySearchResults(max_results=5),
        ]
        assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)

        builder = StateGraph(State)


        # Define nodes: these do the work
        builder.add_node("assistant", self.Assistant(assistant_runnable))
        builder.add_node("action", create_tool_node_with_fallback(tools))
        # Define edges: these determine how the control flow moves
        builder.set_entry_point("assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            # "action" calls one of our tools. END causes the graph to terminate (and respond to the user)
            {"action": "action", END: END},
        )
        builder.add_edge("action", "assistant")

        # The checkpointer lets the graph persist its state
        # this is a complete memory for the entire graph.
        memory = SqliteSaver.from_conn_string(":memory:")
        graph = builder.compile(checkpointer=memory)

        return graph
        # Add saving image of graph

        # Let's create an example conversation a user might have with the assistant
        tutorial_questions = [
            "Hi there, how are you?",
            "Where can I buy pills for headache in Imsouane Morocco, I'm dying?"
        ]

        # Update with the backup file so we can restart from the original place in each section
        #shutil.copy(backup_file, db)
        thread_id = str(uuid.uuid4())

        config = {
            "configurable": {
                "user_name": "Islam",
                # Checkpoints are accessed by thread_id
                "thread_id": thread_id,
            }
        }


    """        _printed = set()
            for question in tutorial_questions:
                events = graph.stream(
                    {"messages": ("user", question)}, config, stream_mode="values"
                )
                for event in events:
                    _print_event(event, _printed)"""

    """
    Problems:
    ==>Doesn't remember user name!
    """