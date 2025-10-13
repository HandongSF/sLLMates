import sqlite3
from typing import Sequence, Dict, List, Optional
from typing_extensions import Annotated, TypedDict

from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from src.config import SQLITE_DB_FILE
from src.core.llm import LLM
from src.core.trimmer import Trimmer
from src.core.tools import TOOLS_LIST
from src.utils.parsers import makeJSONToToolCall




class State(TypedDict):
    variables: Dict[str, str]
    system_prompt: str
    history: Annotated[Sequence[BaseMessage], add_messages]
    messages: Optional[List[BaseMessage]]
    tools_result: Optional[List[ToolMessage]]
    query: HumanMessage
    final_answer: Optional[AIMessage]


class LangChainAgent:
    def __init__(self):
        self.llm = LLM.get_llm()
        self.trimmer = Trimmer.get_trimmer()
        self.tools = ToolNode(TOOLS_LIST)
        self.app = self.create_workflow()

    def query_or_respond(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + state["history"] + [state["query"]])

        llm_with_tools = self.llm.bind_tools(TOOLS_LIST)

        response = llm_with_tools.invoke(trimmed_messages)

        content = response.content.strip()
        if content.startswith("<tool_call>"):
            tools_call = AIMessage(content = "", tool_calls = makeJSONToToolCall(content))
            return {
                "variables": state["variables"],
                "system_prompt": state["system_prompt"],
                "messages": [tools_call],
                "tools_result": None,
                "query": state["query"],
                "final_answer": None
            }

        add_messages = [state["query"]] + [response]

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "history": add_messages,
            "messages": None,
            "tools_result": None,
            "query": state["query"],
            "final_answer": response
        }

    def check_for_tools(self, state: State):
        if state.get("messages"):
            return "tools"
        else:
            return END

    def run_tools_and_pass_through_state(self, state: State):
        tools_result = self.tools.invoke(state["messages"])

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "messages": state["messages"],
            "tools_result": tools_result,
            "query": state["query"],
            "final_answer": None
        }

    def generate(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + state["tools_result"] + [state["query"]])

        response = self.llm.invoke(trimmed_messages)

        add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "history": add_messages,
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "query": state["query"],
            "final_answer": response
        }
    
    def create_workflow(self):
        workflow = StateGraph(state_schema = State)
    
        workflow.add_node("query_or_respond", self.query_or_respond)
        workflow.add_node("run_tools_and_pass_through_state", self.run_tools_and_pass_through_state)
        workflow.add_node("generate", self.generate)
        
        workflow.add_edge(START, "query_or_respond")
        workflow.add_conditional_edges("query_or_respond", self.check_for_tools, {END: END, "tools": "run_tools_and_pass_through_state"})
        workflow.add_edge("run_tools_and_pass_through_state", "generate")
        workflow.add_edge("generate", END)
        
        memory = SqliteSaver(conn=sqlite3.connect(SQLITE_DB_FILE, check_same_thread = False))
        
        return workflow.compile(checkpointer = memory)