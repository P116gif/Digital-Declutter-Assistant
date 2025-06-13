import shutil
import os
import json 
from dotenv import load_dotenv


from langchain_core.tools  import tool
from langchain_ollama import ChatOllama

load_dotenv()

@tool
def _move_file(source: str, destination: str) -> str:
    """
    This tool is used to move files from the source folder to the destination folder, while also handling
    conflict resolution.

    Args:
        source: the current file path of the file
        destination: the location to move the file into

    Returns:
        new_destination: the new file path of the file
    """


    #if the directory does not exist, then make one, else let it be
    os.makedirs(destination, exist_ok=True)

    filename = os.path.basename(source)

    base, ext = os.path.splitext(filename)
    count =  1
    new_destination = destination

    while os.path.exists(new_destination):
        new_filename = f"{base}_{count}{ext}"
        new_destination = os.path.join(destination, new_filename)
        count += 1

    shutil.move(source, new_destination)
    
    return new_destination
    

#defining the llm model
llm = ChatOllama(
    model="qwen3",
    temperature=0,
    base_url="http://localhost:11434"
    )

"""
OR
llm = ChatGroq(
    model = "qwen/qwen3-32b",
    temperature=0,
)
"""

tools = [_move_file]
llm_with_tool = llm.bind_tools(tools)



from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
        This state ensures the messages don't override each other and instead are appended into an ever increasing context to be passed to the llm
    """
    messages: Annotated[list, add_messages]


#creating a tool node using _move_file func
tool_node = ToolNode(tools=tools)


#the chatbot itself
def chatbot(state: State):
    """
        The chatbot function that calls the llm and returns the messages
    """
    return {"messages": [llm_with_tool.invoke(state["messages"])]}


#building the graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")


graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    """
        This function streams the AI output onto the terminal where you're running it
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


"""
 The entry point of this file which runs the graph
"""
while True:      
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)
    
