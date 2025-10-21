import sqlite3
from typing import Sequence, Dict, List, Optional
from typing_extensions import Annotated, TypedDict
import langchain
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from langchain_community.chat_models import ChatLlamaCpp

from src.config import SQLITE_DB_FILE, USING_LLAMA, LLMConfig, TrimmerConfig
from src.core.tools import TOOL_LIST
from src.core.templete import convert_messages_to_text_format_llama3
from src.utils.parsers import parse_llm_output
from src.chat_models.ChatLlamaCpp_new import ChatLlamaCpp_new
from src.chat_models.Llama_new import Llama_new


langchain.debug = True


class State(TypedDict):
    """요약 추가 예정"""

    variables: Dict[str, str]
    """요약 추가 예정"""

    system_prompt: str
    """요약 추가 예정"""

    history: Annotated[Sequence[BaseMessage], add_messages]
    """요약 추가 예정"""

    messages: Optional[List[BaseMessage]]
    """요약 추가 예정"""

    tools_result: Optional[List[ToolMessage]]
    """요약 추가 예정"""

    query: HumanMessage
    """요약 추가 예정"""

    final_answer: Optional[AIMessage]
    """요약 추가 예정"""


class LangChainAgent:
    """요약 추가 예정

    디테일 추가 예정
    """

    llm: ChatLlamaCpp
    """요약 추가 예정"""

    trimmer: any
    """요약 추가 예정"""

    #tool_index: ToolIndex
    """요약 추가 예정"""

    tools: ToolNode
    """요약 추가 예정"""

    app: any
    """요약 추가 예정"""

    def __init__(self):
        if USING_LLAMA:
            self.llm = Llama_new(
                model_path = LLMConfig.model_path,
                n_ctx = LLMConfig.n_ctx,
                f16_kv = LLMConfig.f16_kv,
                use_mlock = LLMConfig.use_mlock,
                n_batch = LLMConfig.n_batch,
                n_gpu_layers = LLMConfig.n_gpu_layers,
                max_tokens = LLMConfig.max_tokens,
                temperature = LLMConfig.temperature,
                top_p = LLMConfig.top_p,
                stop = LLMConfig.stop,
                top_k = LLMConfig.top_k,
                use_mmap = LLMConfig.use_mmap,
                verbose = LLMConfig.verbose,
                main_gpu = LLMConfig.model_kwargs["main_gpu"],
                tensor_split = LLMConfig.model_kwargs["tensor_split"], 
                min_p = LLMConfig.model_kwargs["min_p"],
            )
        else:
            self.llm = ChatLlamaCpp_new(
                model_path = LLMConfig.model_path,
                n_ctx = LLMConfig.n_ctx,
                f16_kv = LLMConfig.f16_kv,
                use_mlock = LLMConfig.use_mlock,
                n_batch = LLMConfig.n_batch,
                n_gpu_layers = LLMConfig.n_gpu_layers,
                max_tokens = LLMConfig.max_tokens,
                temperature = LLMConfig.temperature,
                top_p = LLMConfig.top_p,
                stop = LLMConfig.stop,
                top_k = LLMConfig.top_k,
                use_mmap = LLMConfig.use_mmap,
                model_kwargs = LLMConfig.model_kwargs,
                verbose = LLMConfig.verbose,
            )

        self.trimmer = trim_messages(
            max_tokens = TrimmerConfig.max_tokens,
            strategy = TrimmerConfig.strategy,
            token_counter = self.llm,
            include_system = TrimmerConfig.include_system,
            allow_partial = TrimmerConfig.allow_partial,
            start_on = TrimmerConfig.start_on,
        )

        self.tools = ToolNode(TOOL_LIST)

        self.app = self.create_workflow()

    def query_or_respond(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + state["history"] + [state["query"]])

        if USING_LLAMA:
            response_data = self.llm.create_completion(
                prompt = convert_messages_to_text_format_llama3(trimmed_messages, TOOL_LIST),
                max_tokens = LLMConfig.max_tokens,
                temperature = LLMConfig.temperature,
                top_p = LLMConfig.top_p,
                min_p = LLMConfig.model_kwargs["min_p"],
                stop = LLMConfig.stop,
                top_k = LLMConfig.top_k,      
            )
            print("모델 답변: query_or_respond: " + repr(response_data))
            response = parse_llm_output(response_data)
            print("모델 답변 가공: query_or_respond: " + repr(response))
        else:
            llm_with_tools = self.llm.bind_tools(TOOL_LIST)
            response = llm_with_tools.invoke(trimmed_messages)

        if response.tool_calls:
            return {
                "variables": state["variables"],
                "system_prompt": state["system_prompt"],
                "messages": [response],
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

        if USING_LLAMA:
            response_data = self.llm.create_completion(
                prompt = convert_messages_to_text_format_llama3(trimmed_messages),
                max_tokens = LLMConfig.max_tokens,
                temperature = LLMConfig.temperature,
                top_p = LLMConfig.top_p,
                min_p = LLMConfig.model_kwargs["min_p"],
                stop = LLMConfig.stop,
                top_k = LLMConfig.top_k,      
            )
            print("모델 답변: generate: " + repr(response_data))
            response = parse_llm_output(response_data)
            print("모델 답변 가공: generate: " + repr(response))
        else:
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