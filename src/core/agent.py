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

from src.config import SQLITE_DB_FILE, USING_LLAMA, LLMConfig, TrimmerConfig, BIO_EXPLANATION_PROMPT, BIO_PROMPT
from src.core.tools import TOOL_LIST
from src.core.templete import convert_messages_to_text_format_llama3
from src.utils.parsers import parse_llm_output
from src.chat_models.ChatLlamaCpp_new import ChatLlamaCpp_new
from src.chat_models.Llama_new import Llama_new
from src.db.bio_metadata import search_similar_bios, save_or_update_bio
from src.tool.bio_manager import BioManager
import time

langchain.debug = True


class State(TypedDict):
    """LangGraph에서 사용하는 에이전트 전체 상태(State).

    사용자 입력, 시스템 프롬프트, 대화 히스토리, tool 실행 결과,
    bio memory 검색 결과, 최종 응답을 모두 포함한다.
    """

    variables: Dict[str, str]
    """시스템 프롬프트 템플릿에 주입될 변수들"""

    system_prompt: str
    """LLM에 전달되는 기본 시스템 프롬프트 (format string)"""

    history: Annotated[Sequence[BaseMessage], add_messages]
    """tool 호출이 없는 순수 대화 히스토리 (누적됨)"""

    messages: Optional[List[BaseMessage]]
    """tool 호출이 포함된 임시 메시지 (tool 실행용)"""

    tools_result: Optional[List[ToolMessage]]
    """tool 실행 결과 메시지"""

    bio_result: Optional[str]
    """bio memory 검색 결과를 시스템 컨텍스트로 변환한 문자열"""

    query: HumanMessage
    """현재 사용자 입력"""

    final_answer: Optional[AIMessage]
    """최종 LLM 응답"""


class LangChainAgent:
    """sLLM 기반 챗봇 에이전트.

    - LangGraph를 사용해 상태 기반 워크플로우 구성
    - Tool calling 지원
    - Bio memory 검색 및 저장(RAG-like memory)
    - llama.cpp 기반 sLLM 실행
    """

    llm: ChatLlamaCpp
    """llama.cpp 기반 sLLM 모델 래퍼"""

    trimmer: any
    """대화 토큰 수를 제한하기 위한 메시지 트리머"""

    tools: ToolNode
    """LangGraph에서 실행되는 tool 노드"""

    app: any
    """컴파일된 LangGraph 애플리케이션"""

    bio_manager: BioManager
    """bio memory 추출 및 중요도 판단 로직"""

    def __init__(self):
        if USING_LLAMA:
            self.llm = Llama_new(
                model_path = LLMConfig.model_path,
                n_ctx = LLMConfig.n_ctx,
                f16_kv = LLMConfig.f16_kv,
                use_mlock = LLMConfig.use_mlock,
                n_batch = LLMConfig.n_batch,
                n_gpu_layers = LLMConfig.n_gpu_layers,
                use_mmap = LLMConfig.use_mmap,
                verbose = LLMConfig.verbose,
                main_gpu = LLMConfig.model_kwargs["main_gpu"],
                tensor_split = LLMConfig.model_kwargs["tensor_split"], 
                flash_attn = LLMConfig.model_kwargs["flash_attn"],
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

        self.bio_manager = BioManager()

    def retrieve_bio_memory(self, state: State):
        bio_dict = search_similar_bios(state["query"].content, 5)

        bio_result = BIO_EXPLANATION_PROMPT

        if bio_dict:
            for bio in bio_dict[:10]:
                bio_result += f"-{bio["document"]}\n"
        else:
            bio_result = ""

        return {
            "bio_result":bio_result
        }
    
    def extract_and_save_bio_memory(self, state:State):
        start = time.time()
        bio_prompt = BIO_PROMPT

        trimmed_messages = self.trimmer.invoke([SystemMessage(bio_prompt)] + [state["query"]])

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
            response = parse_llm_output(response_data)
            print("extract_and_save_bio_memory 결과: " + repr(response))
        else:
            response = self.llm.invoke(trimmed_messages)

        if response:
            bio_list = self.bio_manager.extract_bio_with_importance(response.content)
            if bio_list:
                save_or_update_bio(bio_list)
            #state["final_answer"].content = self.bio_manager.clean_bio_tags(state["final_answer"].content)
        end = time.time()
        print(f"extract_and_save_bio_memory 실행 시간: {end - start:.5f}초")
        return 

    def query_or_respond(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + [state["bio_result"]] + conversation_messages + [state["query"]])

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
            return "no_tool"

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

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + [state["bio_result"]] + conversation_messages + state["tools_result"] + [state["query"]])

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
    
        workflow.add_node("retrieve_bio_memory", self.retrieve_bio_memory)
        workflow.add_node("query_or_respond", self.query_or_respond)
        workflow.add_node("run_tools_and_pass_through_state", self.run_tools_and_pass_through_state)
        workflow.add_node("generate", self.generate)
        workflow.add_node("extract_and_save_bio_memory", self.extract_and_save_bio_memory)
        
        workflow.add_edge(START, "retrieve_bio_memory")
        workflow.add_edge("retrieve_bio_memory", "query_or_respond")
        workflow.add_conditional_edges("query_or_respond", self.check_for_tools, {"no_tool": "extract_and_save_bio_memory", "tools": "run_tools_and_pass_through_state"})
        workflow.add_edge("run_tools_and_pass_through_state", "generate")
        workflow.add_edge("generate", "extract_and_save_bio_memory")
        workflow.add_edge("extract_and_save_bio_memory", END)
        
        memory = SqliteSaver(conn=sqlite3.connect(SQLITE_DB_FILE, check_same_thread = False))
        
        return workflow.compile(checkpointer = memory)