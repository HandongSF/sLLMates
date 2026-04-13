import sys
import sqlite3
import time
from pprint import pprint
from datetime import datetime
import importlib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Sequence, Dict, List, Optional, Tuple, Annotated, Union
from typing_extensions import Annotated, TypedDict
from llama_cpp import Llama, LlamaState
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from langchain_core.utils.function_calling import convert_to_openai_tool
import langchain
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.messages import convert_to_openai_messages
from langchain_core.tools import BaseTool
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.types import StreamWriter
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from langchain_community.chat_models import ChatLlamaCpp
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import SELECTED_CONFIG_FILE, SQLITE_DB_FILE
from src.db.vector_store import ChromaDBManager
from src.db.bio_metadata import BioMetadata
from src.core.parsers import parse_llm_output, convert_messages_to_llama3_messages, parse_bio_with_importance, parse_query_for_bio


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

    branch_name: str
    """사용할 branch 이름"""

    classifier_result: Optional[str]
    """classifier 결과"""

    messages: Optional[List[BaseMessage]]
    """tool 호출이 포함된 임시 메시지 (tool 실행용)"""

    tools_result: Optional[List[ToolMessage]]
    """tool 실행 결과 메시지"""

    bio_result: Optional[Tuple[str, str]]
    """bio memory 검색 결과를 시스템 컨텍스트로 변환한 문자열"""

    upcoming_thread_id: Optional[str]
    """현재 대화 스레드 ID (대화창이 바뀔 때 bio 추출 실행)"""

    query: HumanMessage
    """현재 사용자 입력"""

    final_answer: Optional[Union[AIMessage, str]]
    """최종 LLM 응답"""

class ChatAgent:
    """sLLM 기반 채팅 에이전트.

    - LangGraph를 사용해 상태 기반 워크플로우 구성
    - Tool calling 지원
    - Bio memory 검색 및 저장(RAG-like memory)
    - llama.cpp 기반 sLLM 실행
    """

    config: any
    """모델 설정 파일에서 가져온 딕셔너리값들이 들어있는 변수"""

    llm: Llama
    """llama.cpp의 llm 클래스"""

    kv_cache_snapshot: Optional[LlamaState] = None
    """llama.cpp의 키-값 캐시 스냅샷 (바이너리 형태)"""

    current_thread_id: Optional[str]

    bio_extraction_buffer: Optional[Dict[str, any]]
    """bio extraction을 위해 query들을 모아놓는 버퍼, token count도 포함"""

    formatter: any

    trimmer: any
    """대화 토큰 수를 제한하기 위한 메시지 트리머"""

    classifier_tokenizer: any

    classifier_llm: any

    chroma_db_manager: any

    bio_metadata: any

    tool_list: any

    tools: ToolNode
    """LangGraph에서 실행되는 tool 노드"""

    app: any
    """컴파일된 LangGraph 애플리케이션"""

    def __init__(self):
        # config 설정 변수들 가져오기
        self.config = self.load_chat_model_config()

        # formatter 선언
        if self.config.get("USE_CUSTOM_CHAT_HANDLER", False):
            if self.config.get("CUSTOM_CHAT_TEMPLATE", "") and self.config.get("FORMATTER_CONFIG", "").get("eos_token", "") and self.config.get("FORMATTER_CONFIG", "").get("bos_token", ""):
                self.formatter = Jinja2ChatFormatter(
                    template = self.config["CUSTOM_CHAT_TEMPLATE"],
                    eos_token = self.config["FORMATTER_CONFIG"]["eos_token"],
                    bos_token = self.config["FORMATTER_CONFIG"]["bos_token"],
                )
            else:
                self.formatter = None
        else:
            self.formatter = None

        # 채팅 모델 선언
        if self.config.get("CHAT_MODEL_CONFIG", {}):
            self.llm = Llama(
                model_path = self.config["CHAT_MODEL_CONFIG"].get("model_path", ""),
                # Model Params
                n_gpu_layers = self.config["CHAT_MODEL_CONFIG"].get("n_gpu_layers", 0),
                main_gpu = self.config["CHAT_MODEL_CONFIG"].get("main_gpu", 0),
                tensor_split = self.config["CHAT_MODEL_CONFIG"].get("tensor_split", None), 
                use_mmap = self.config["CHAT_MODEL_CONFIG"].get("use_mmap", True),
                use_mlock = self.config["CHAT_MODEL_CONFIG"].get("use_mlock", False),
                # Context Params
                n_ctx = self.config["CHAT_MODEL_CONFIG"].get("n_ctx", 512),
                n_batch = self.config["CHAT_MODEL_CONFIG"].get("n_batch", 512),
                flash_attn = self.config["CHAT_MODEL_CONFIG"].get("flash_attn", False),
                # Chat Format Params
                chat_handler = self.formatter.to_chat_handler() if self.formatter is not None else None,
                # Misc
                verbose = self.config["CHAT_MODEL_CONFIG"].get("verbose", True),
            )
        else:
            print("에러: CHAT_MODEL_CONFIG 딕셔너리가 없음")
            sys.exit(1)

        # 분류 모델 선언
        if self.config.get("CLASSIFIER_MODEL_CONFIG", {}):
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.config["CLASSIFIER_MODEL_CONFIG"].get("model_path", ""))
            self.classifier_llm = AutoModelForSequenceClassification.from_pretrained(self.config["CLASSIFIER_MODEL_CONFIG"].get("model_path", ""))
        else:
            self.classifier_tokenizer = None
            self.classifier_llm = None

        # 트리머 선언
        if self.config.get("TRIMMER_CONFIG", {}):
            self.trimmer = trim_messages(
                max_tokens = self.config["TRIMMER_CONFIG"].get("max_tokens", 256),
                strategy = self.config["TRIMMER_CONFIG"].get("strategy", "last"),
                token_counter = self.get_num_tokens_from_messages,
                include_system = self.config["TRIMMER_CONFIG"].get("include_system", True),
                allow_partial = self.config["TRIMMER_CONFIG"].get("allow_partial", False),
                start_on = self.config["TRIMMER_CONFIG"].get("start_on", "human"),
            )
        else:
            print("에러: TRIMMER_CONFIG 딕셔너리가 없음")
            sys.exit(1)

        # 툴 함수들 선언

        self.chroma_db_manager = ChromaDBManager(self.config)

        if not self.config.get("RAG_CONFIG", {}):
            print("에러: RAG_CONFIG 딕셔너리가 없음")
            sys.exit(1)

        @tool(response_format="content_and_artifact")
        def retrieve(
            query: Annotated[str, "A search query composed of the essential keywords from the user's question. For example: 'Tell me the name of the largest bird' -> 'the largest bird'"]
        ):
            """You have the tool `retrieve`. Use `retrieve` in the following circumstances:\n - User is asking about some term you are totally unfamiliar with (it might be new).\n - User explicitly asks you to browse or provide links to references.\n\n Given a query that requires retrieval, you call the retrieve function to get a list of results."
            """

            if query == "__NONE__":
                return "No results found.", []

            retrieved_docs = self.chroma_db_manager.get_doc_store().similarity_search(query, k = self.config["RAG_CONFIG"].get("retrieval_k", 5))

            if not retrieved_docs:
                return "No results found.", []

            serialized = "\n\n".join(
                (f"{doc.page_content}")
                for doc in retrieved_docs
            )

            return serialized, retrieved_docs
        
        self.tool_list = [retrieve, ]

        self.tools = ToolNode(self.tool_list)

        self.bio_metadata = BioMetadata(self.chroma_db_manager.get_bio_store())

        self.bio_cleanup_scheduler()

        self.bio_extraction_buffer= {
                "queries": [],
                "token_count": 0,
                "query_count": 0
            }

        self.current_thread_id = None

        # app 선언
        self.app = self.create_workflow()

    # 세팅 함수

    def load_chat_model_config(self):
        model_module_name = SELECTED_CONFIG_FILE

        full_model_module_path = f"src.configs.{model_module_name}"

        if full_model_module_path.endswith('.py'):
            full_model_module_path = full_model_module_path[:-3]

        try:
            module = importlib.import_module(full_model_module_path)
        
            return getattr(module, "CONFIG", {})

        except ModuleNotFoundError:
            print(f"에러: {model_module_name} 파일을 찾을 수 없습니다.")
            raise
        except AttributeError:
            print(f"에러: {model_module_name} 안에 CONFIG 딕셔너리가 없습니다.")
            raise

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        total_tokens = 0
        for message in messages:
            content_str = ""

            if isinstance(message.content, str):
                content_str = message.content
            elif isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_str += part.get("text", "")

            if content_str:
                try:
                    message_bytes = content_str.encode("utf-8")
                    tokens = self.llm.tokenize(message_bytes)
                    total_tokens += len(tokens)
                except Exception as e:
                    print(f"Warning: Could not tokenize message content: {e}")
                    total_tokens += len(content_str) // 4
        
        return total_tokens
    
    def get_num_tokens_from_query(self, query: HumanMessage) -> int:
        content_str = query.content

        if content_str:
            try:
                message_bytes = content_str.encode("utf-8")
                tokens = self.llm.tokenize(message_bytes)
                return len(tokens)
            except Exception as e:
                print(f"Warning: Could not tokenize query content: {e}")
                return len(content_str) // 4
        
        return 0
    
    def bio_cleanup_scheduler(self):
        """기한이 만료된 Bio 메모리를 정리 스케줄러 함수 """
        self.scheduler = BackgroundScheduler()

        # Trigger the event everyday at 00:00/midnight 
        trigger = CronTrigger(hour=0, minute=0)

        self.scheduler.add_job(
            self.bio_metadata.cleanup_expired_bio_memories,
            trigger=trigger,
            id="daily_memory_cleanup",
            name="Delete expired bio memories every midnight",
            replace_existing=True
        )

        self.scheduler.start()
        print("[Bio Scheduler] 매일 자정 기한이 만료된 Bio memory가 자동 삭제됩니다.")
    
    # 노드 함수

    def router(self, state: State):
        if state.get("branch_name") == "default":
            return "default"
        elif state.get("branch_name") == "tools":
            return "tools"
        elif state.get("branch_name") == "classifier":
            return "classifier"
        elif state.get("branch_name") == "bio":
            return "bio"
        elif state.get("branch_name") == "stream":
            return "stream"
        elif state.get("branch_name") == "fusion":
            return "fusion"
        elif state.get("branch_name") == "fusiontool":
            return "fusiontool"
        elif state.get("branch_name") == "fusiontool_v2":
            return "fusiontool_v2"
        else:
            return "default"
        
    # branch name: default

    def default_generate(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        # pprint(openai_formatted_trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            # print(full_prompt)

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            # pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            # print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            # pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        add_messages = [state["query"]] + [response]

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "history": add_messages,
            "branch_name": state["branch_name"],
            "messages": None,
            "tools_result": None,
            "query": state["query"],
            "final_answer": response
        }

    # branch name: tools

    def tools_query_or_respond(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        openai_formatted_tools = [convert_to_openai_tool(tool) for tool in self.tool_list]

        pprint(openai_formatted_trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
            ).prompt

            print(full_prompt)

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)
 
        if response.tool_calls:
            return {
                "variables": state["variables"],
                "system_prompt": state["system_prompt"],
                "branch_name": state["branch_name"],
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
            "branch_name": state["branch_name"],
            "messages": None,
            "tools_result": None,
            "query": state["query"],
            "final_answer": response
        }

    def tools_check_for_tools(self, state: State):
        if state.get("messages"):
            return "tools"
        else:
            return "no_tool"

    def tools_run_tools_and_pass_through_state(self, state: State):
        tools_result = self.tools.invoke(state["messages"])

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "messages": state["messages"],
            "tools_result": tools_result,
            "query": state["query"],
            "final_answer": None
        }

    def tools_generate(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + state["tools_result"] + [state["query"]])

        openai_formatted_trimmed_messages = convert_messages_to_llama3_messages(trimmed_messages)

        pprint(openai_formatted_trimmed_messages)

        if self.formatter:
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            print(full_prompt)

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)

        add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "history": add_messages,
            "branch_name": state["branch_name"],
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "query": state["query"],
            "final_answer": response
        }

    # branch name: classifier

    def classifier_check_thinking(self, state: State):
        if self.classifier_tokenizer == None or self.classifier_llm == None:
            print("에러: classifier_tokenizer 또는 classifier_llm 없음")
            sys.exit(1)

        id2label = {0: "Non-thinking", 1: "Thinking"}

        print(">>>query: " + state["query"].content)

        raw_tokens = self.classifier_tokenizer.encode(state["query"].content, add_special_tokens=False)

        if len(raw_tokens) > 510:
            print(f"Query가 너무 길어 512 토큰에 맞게 자르기 시행 ({len(raw_tokens)} -> 510)")
            raw_tokens = raw_tokens[:128] + raw_tokens[-382:]

        input_ids = [self.classifier_tokenizer.cls_token_id] + raw_tokens + [self.classifier_tokenizer.sep_token_id]

        inputs = {
            "input_ids": torch.tensor([input_ids]).to(self.classifier_llm.device),
            "attention_mask": torch.ones((1, len(input_ids))).to(self.classifier_llm.device)
        }

        with torch.no_grad():
            outputs = self.classifier_llm(**inputs)
    
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()

        label_name = id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")

        print(">>Label_name: " + label_name)

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": label_name,
            "messages": None,
            "tools_result": None,
            "query": state["query"],
            "final_answer": None
        }

    def classifier_query_or_respond(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        openai_formatted_tools = [convert_to_openai_tool(tool) for tool in self.tool_list]

        #pprint(openai_formatted_trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)
            print('\n\n\n\n')

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)
            print('\n\n\n\n')

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)
        print('\n\n\n\n')
 
        if response.tool_calls:
            return {
                "variables": state["variables"],
                "system_prompt": state["system_prompt"],
                "branch_name": state["branch_name"],
                "classifier_result": state["classifier_result"],
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
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": None,
            "tools_result": None,
            "query": state["query"],
            "final_answer": response
        }

    def classifier_check_for_tools(self, state: State):
        if state.get("messages"):
            return "tools"
        else:
            return "no_tool"

    def classifier_run_tools_and_pass_through_state(self, state: State):
        tools_result = self.tools.invoke(state["messages"])

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": tools_result,
            "query": state["query"],
            "final_answer": None
        }

    def classifier_generate(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + state["tools_result"] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        pprint(openai_formatted_trimmed_messages)

        if self.formatter:
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)

        add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "history": add_messages,
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "query": state["query"],
            "final_answer": response
        }

    # branch name: bio

    def bio_retrieve_bio_memory(self, state: State):
        if self.kv_cache_snapshot is not None:
            start_time = time.time()
            self.llm.load_state(self.kv_cache_snapshot) # KV 캐시 스냅샷 로드
            self.kv_cache_snapshot = None # 로드 후 스냅샷 초기화
            print("Bio Memory 실행 후, 이전 대화 KV 캐시 스냅샷이 로드되었습니다.")
            end_time = time.time()
            print(f"KV 캐시 스냅샷 로드에 걸린 시간: {end_time - start_time:.2f} seconds")


        top_k = self.config["BIO_CONFIG"].get("top_k", 5)
        threshold = self.config["BIO_CONFIG"].get("retrieval_threshold_kor", 1.15)

        core_data = self.bio_metadata.get_bio_chroma_collection()._collection.get(
        where={"is_core": True}  
        )

        core_docs = core_data.get("documents", [])

        if core_docs:
            bio_core_result = self.config["CORE_BIO_EXPLANATION_PROMPT"]
            for doc in core_docs:
                bio_core_result += f"- {doc}\n"
        else:
            bio_core_result = ""

        parse_query_for_bio_result = parse_query_for_bio(state["query"].content)

        vector = self.bio_metadata.get_embedding_function().embed_query(parse_query_for_bio_result)
        retrieved_bio = self.bio_metadata.get_bio_chroma_collection()._collection.query(
            query_embeddings=[vector],
            n_results = top_k,
            include = ["documents", "metadatas", "distances"]
        )

        bio_general_result = self.config["BIO_EXPLANATION_PROMPT_KOR"]
        general_docs = []

        if retrieved_bio['documents'] and retrieved_bio['documents'][0]:
            for i in range(len(retrieved_bio['documents'][0])):
                distance = retrieved_bio['distances'][0][i]
                content = retrieved_bio['documents'][0][i]
                metadata = retrieved_bio["metadatas"][0][i]

                if distance <= threshold and not metadata.get("is_core", False):
                    print(f"Bio memory retrieved with distance {distance}: {content}")
                    general_docs.append(content)

        if general_docs:
            for doc in general_docs:
                bio_general_result += f"- {doc}\n"
        else:
            bio_general_result = ""

        return {
            "bio_result": [bio_core_result, bio_general_result]
        }

    def bio_generate(self, state: State):
        start_time_for_generate = time.time()
        
        filled_system_prompt = state["system_prompt"].format(**state["variables"]) + state["bio_result"][0]

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [ToolMessage(content=state["bio_result"][1], tool_call_id="temp")] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        pprint(openai_formatted_trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        add_messages = [state["query"]] + [response]


        self.bio_extraction_buffer["queries"].append(state["query"].content)
        self.bio_extraction_buffer["token_count"] += self.get_num_tokens_from_query(state["query"])
        self.bio_extraction_buffer["query_count"] += 1
        print(f"현재 추가된 Bio 추출 버퍼 쿼리: {state['query'].content}")
        print(f"현재 추가된 쿼리 토큰 수: {self.get_num_tokens_from_query(state['query'])}")
        print(f"현재 Bio 추출 버퍼에 쌓인 토큰 수: {self.bio_extraction_buffer['token_count']}")
        print(f"현재 Bio 추출 버퍼에 쌓인 쿼리 수: {self.bio_extraction_buffer['query_count']}")

        end_time_for_generate = time.time()
        print(f"Bio generate 실행 시간: {end_time_for_generate - start_time_for_generate:.5f}초")

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "history": add_messages,
            "branch_name": state["branch_name"],
            "messages": None,
            "tools_result": None,
            "query": state["query"],
            "final_answer": response
        }
    
    def bio_check_for_bio_extraction(self, state: State):
        upcoming_id = state.get("upcoming_thread_id")
        buffer = self.bio_extraction_buffer
        threshold = self.config.get("BIO_CONFIG", {}).get("extraction_token_threshold", 512)
        
        if buffer.get("query_count", 0) == 0:
            return "skip_bio_extraction"

        # 대화창이 바뀌었을 때 (upcoming_thread_id가 현재 thread_id와 다르고, 버퍼에 쌓인 쿼리가 2개 이상일 때)
        is_new_thread = upcoming_id and upcoming_id != self.current_thread_id and buffer.get("query_count", 0) > 1
        
        # 버퍼에 쌓인 토큰 수가 임계치 이상일 경우
        is_threshold_met = buffer.get("token_count", 0) >= threshold

        if is_new_thread:
            self.current_thread_id = upcoming_id # ID 업데이트
            print("대화창이 변경되어 Bio 추출을 시작합니다.")
            return "extract_bio"
            
        if is_threshold_met:
            print("버퍼에 쌓인 토큰 수가 임계치를 초과하여 Bio 추출을 시작합니다.")
            return "extract_bio"

        self.current_thread_id = upcoming_id # ID 업데이트 (대화창이 바뀌지 않았더라도 ID는 업데이트)

        return "skip_bio_extraction"

    def bio_extract_and_save_bio_memory(self, state:State):
        start = time.time()

        self.kv_cache_snapshot = self.llm.save_state() # KV 캐시 스냅샷 저장
        print("Bio Memory 실행 전, 현재 대화 KV 캐시 스냅샷이 저장되었습니다.")

        bio_extraction_prompt = self.config["BIO_EXTRACTION_PROMPT_KOR"]
        buffer_messages = [HumanMessage(content=q) for q in self.bio_extraction_buffer["queries"]]

        trimmed_messages = self.trimmer.invoke([SystemMessage(bio_extraction_prompt)] + buffer_messages)

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            pprint(full_prompt)

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)
        
        if response:
            bio_list = parse_bio_with_importance(response.content)
            if bio_list:
                self.bio_metadata.save_or_update_bio(bio_list)

        end = time.time()

        print(f"extract_and_save_bio_memory 실행 시간: {end - start:.5f}초")
        self.bio_extraction_buffer= {
                "queries": [],
                "token_count": 0,
                "query_count": 0
        }

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            #"classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": None
        }
    
    # branch name: stream

    def stream_check_thinking(self, state: State):
        if self.classifier_tokenizer == None or self.classifier_llm == None:
            print("에러: classifier_tokenizer 또는 classifier_llm 없음")
            sys.exit(1)

        id2label = {0: "Non-thinking", 1: "Thinking"}

        # print(">>>query: " + state["query"].content)

        raw_tokens = self.classifier_tokenizer.encode(state["query"].content, add_special_tokens=False)

        if len(raw_tokens) > 510:
            print(f"Query가 너무 길어 512 토큰에 맞게 자르기 시행 ({len(raw_tokens)} -> 510)")
            raw_tokens = raw_tokens[:128] + raw_tokens[-382:]

        input_ids = [self.classifier_tokenizer.cls_token_id] + raw_tokens + [self.classifier_tokenizer.sep_token_id]

        inputs = {
            "input_ids": torch.tensor([input_ids]).to(self.classifier_llm.device),
            "attention_mask": torch.ones((1, len(input_ids))).to(self.classifier_llm.device)
        }

        with torch.no_grad():
            outputs = self.classifier_llm(**inputs)
    
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()

        label_name = id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")

        # print("classifier_result = " + label_name)

        return {
            "classifier_result": label_name,
        }

    def stream_query_or_respond(self, state: State, writer: StreamWriter):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        openai_formatted_tools = [convert_to_openai_tool(tool) for tool in self.tool_list]

        #pprint(openai_formatted_trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            # print(full_prompt)
            # print('\n\n\n\n')

            response_generator = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),   
                stream = True, 
            )

            text_output = ""
            for chunk in response_generator:
                # print("stream_query_or_respond: 토큰 생성 중...")
                token = chunk['choices'][0]['text']
                text_output += token
                writer({"final_answer": token})
                #yield {"final_answer": token}

            # print("text_output >>>")
            # pprint(text_output)
            # print('\n\n\n\n')
        else:
            print("formatter = None")

            response_generator = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
                stream = True, 
            )

            text_output = ""
            for chunk in response_generator:
                # print("stream_query_or_respond: 토큰 생성 중...")
                token = chunk['choices'][0]['message']['content']
                text_output += token
                writer({"final_answer": token})
                #yield {"final_answer": token}

            # print("text_output >>>")
            # pprint(text_output)
            # print('\n\n\n\n')

        response = parse_llm_output(text_output)

        # print("response >>>")
        # pprint(response)
        # print('\n\n\n\n')
 
        if response.tool_calls:
            return {
                "messages": [response],
            }
        
        add_messages = [state["query"]] + [response]

        return {
            "history": add_messages,
            #"final_answer": response
        }

    def stream_check_for_tools(self, state: State):
        if state.get("messages"):
            return "tools"
        else:
            return "no_tool"

    def stream_run_tools_and_pass_through_state(self, state: State):
        tools_result = self.tools.invoke(state["messages"])

        return {
            "tools_result": tools_result,
        }

    def stream_generate(self, state: State, writer: StreamWriter):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + state["tools_result"] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        # pprint(openai_formatted_trimmed_messages)

        if self.formatter:
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            # print(full_prompt)
            # print('\n\n\n\n')

            response_generator = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),   
                stream = True,  
            )

            text_output = ""
            for chunk in response_generator:
                # print("stream_generate: 토큰 생성 중...")
                token = chunk['choices'][0]['text']
                text_output += token
                writer({"final_answer": token})
                #yield {"final_answer": token}

            # print("text_output >>>")
            # pprint(text_output)
            # print('\n\n\n\n')
        else:
            print("formatter = None")

            response_generator = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
                stream = True, 
            )

            text_output = ""
            for chunk in response_generator:
                # print("stream_generate: 토큰 생성 중...")
                token = chunk['choices'][0]['message']['content']
                text_output += token
                writer({"final_answer": token})
                #yield {"final_answer": token}

            # print("text_output >>>")
            # pprint(text_output)
            # print('\n\n\n\n')
        
        response = parse_llm_output(text_output)

        # print("response >>>")
        # pprint(response)
        # print('\n\n\n\n')

        add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]

        return {
            "history": add_messages,
            #"final_answer": response
        }

    # branch name: fusion

    def fusion_check_thinking(self, state: State):
        if self.classifier_tokenizer == None or self.classifier_llm == None:
            print("에러: classifier_tokenizer 또는 classifier_llm 없음")
            sys.exit(1)

        id2label = {0: "Non-thinking", 1: "Thinking"}

        print(">>>query: " + state["query"].content)

        raw_tokens = self.classifier_tokenizer.encode(state["query"].content, add_special_tokens=False)

        if len(raw_tokens) > 510:
            print(f"Query가 너무 길어 512 토큰에 맞게 자르기 시행 ({len(raw_tokens)} -> 510)")
            raw_tokens = raw_tokens[:128] + raw_tokens[-382:]

        input_ids = [self.classifier_tokenizer.cls_token_id] + raw_tokens + [self.classifier_tokenizer.sep_token_id]

        inputs = {
            "input_ids": torch.tensor([input_ids]).to(self.classifier_llm.device),
            "attention_mask": torch.ones((1, len(input_ids))).to(self.classifier_llm.device)
        }

        with torch.no_grad():
            outputs = self.classifier_llm(**inputs)
    
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()

        label_name = id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")

        print(">>Label_name: " + label_name)

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": label_name,
            "messages": None,
            "tools_result": None,
            "query": state["query"],
            "final_answer": None
        }
    
    def fusion_retrieve_bio_memory(self, state: State):
        top_k = self.config["BIO_CONFIG"].get("top_k", 5)
        threshold = self.config["BIO_CONFIG"].get("retrieval_threshold", 1.0)

        core_data = self.bio_metadata.get_bio_chroma_collection()._collection.get(
        where={"is_core": True}  
        )

        core_docs = core_data.get("documents", [])

        if core_docs:
            bio_core_result = self.config["CORE_BIO_EXPLANATION_PROMPT"]
            for doc in core_docs:
                bio_core_result += f"- {doc}\n"
        else:
            bio_core_result = ""

        vector = self.bio_metadata.get_embedding_function().embed_query(state["query"].content)
        retrieved_bio = self.bio_metadata.get_bio_chroma_collection()._collection.query(
            query_embeddings=[vector],
            n_results = top_k,
            include = ["documents", "metadatas", "distances"]
        )

        bio_general_result = self.config["BIO_EXPLANATION_PROMPT"]
        general_docs = []

        if retrieved_bio['documents'] and retrieved_bio['documents'][0]:
            for i in range(len(retrieved_bio['documents'][0])):
                distance = retrieved_bio['distances'][0][i]
                content = retrieved_bio['documents'][0][i]
                metadata = retrieved_bio["metadatas"][0][i]

                if distance <= threshold and not metadata.get("is_core", False):
                    general_docs.append(content)

        if general_docs:
            for doc in general_docs:
                bio_general_result += f"- {doc}\n"
        else:
            bio_general_result = ""

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": None,
            "tools_result": None,
            "bio_result": [bio_core_result, bio_general_result],
            "query": state["query"],
            "final_answer": None
        }

    def fusion_query_or_respond(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"]) + state["bio_result"][0]

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [ToolMessage(content=state["bio_result"][1], tool_call_id="temp")] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        openai_formatted_tools = [convert_to_openai_tool(tool) for tool in self.tool_list]

        #pprint(openai_formatted_trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)
            print('\n\n\n\n')

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)
            print('\n\n\n\n')

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)
        print('\n\n\n\n')
 
        if response.tool_calls:
            return {
                "variables": state["variables"],
                "system_prompt": state["system_prompt"],
                "branch_name": state["branch_name"],
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
            "branch_name": state["branch_name"],
            "messages": None,
            "tools_result": None,
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": response
        }

    def fusion_check_for_tools(self, state: State):
        if state.get("messages"):
            return "tools"
        else:
            return "no_tool"

    def fusion_run_tools_and_pass_through_state(self, state: State):
        tools_result = self.tools.invoke(state["messages"])

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": tools_result,
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": None
        }

    def fusion_generate(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"]) + state["bio_result"][0]

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]
        
        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + state["tools_result"] + [ToolMessage(content=state["bio_result"][1], tool_call_id="temp")] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        pprint(openai_formatted_trimmed_messages)

        if self.formatter:
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)

        add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "history": add_messages,
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": response
        }

    def fusion_extract_and_save_bio_memory(self, state:State):
        start = time.time()
        bio_extraction_prompt = self.config["BIO_EXTRACTION_PROMPT"]
        trimmed_messages = self.trimmer.invoke([SystemMessage(bio_extraction_prompt)] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)
        
        if response:
            bio_list = parse_bio_with_importance(response.content)
            if bio_list:
                self.bio_metadata.save_or_update_bio(bio_list)
        end = time.time()

        print(f"extract_and_save_bio_memory 실행 시간: {end - start:.5f}초")

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": None
        }

    # branch name: fusiontool

    def fusiontool_check_thinking(self, state: State):
        if self.classifier_tokenizer == None or self.classifier_llm == None:
            print("에러: classifier_tokenizer 또는 classifier_llm 없음")
            sys.exit(1)

        id2label = {0: "Non-thinking", 1: "Thinking"}

        print(">>>query: " + state["query"].content)

        raw_tokens = self.classifier_tokenizer.encode(state["query"].content, add_special_tokens=False)

        if len(raw_tokens) > 510:
            print(f"Query가 너무 길어 512 토큰에 맞게 자르기 시행 ({len(raw_tokens)} -> 510)")
            raw_tokens = raw_tokens[:128] + raw_tokens[-382:]

        input_ids = [self.classifier_tokenizer.cls_token_id] + raw_tokens + [self.classifier_tokenizer.sep_token_id]

        inputs = {
            "input_ids": torch.tensor([input_ids]).to(self.classifier_llm.device),
            "attention_mask": torch.ones((1, len(input_ids))).to(self.classifier_llm.device)
        }

        with torch.no_grad():
            outputs = self.classifier_llm(**inputs)
    
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()

        label_name = id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")

        print(">>Label_name: " + label_name)

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": label_name,
            "messages": None,
            "tools_result": None,
            "query": state["query"],
            "final_answer": None
        }
    
    def fusiontool_retrieve_bio_memory(self, state: State):
        top_k = self.config["BIO_CONFIG"].get("top_k", 5)
        threshold = self.config["BIO_CONFIG"].get("retrieval_threshold", 1.0)

        core_data = self.bio_metadata.get_bio_chroma_collection()._collection.get(
        where={"is_core": True}  
        )

        core_docs = core_data.get("documents", [])

        if core_docs:
            bio_core_result = self.config["CORE_BIO_EXPLANATION_PROMPT"]
            for doc in core_docs:
                bio_core_result += f"- {doc}\n"
        else:
            bio_core_result = ""

        vector = self.bio_metadata.get_embedding_function().embed_query(state["query"].content)
        retrieved_bio = self.bio_metadata.get_bio_chroma_collection()._collection.query(
            query_embeddings=[vector],
            n_results = top_k,
            include = ["documents", "metadatas", "distances"]
        )

        bio_general_result = self.config["BIO_EXPLANATION_PROMPT"]
        general_docs = []

        if retrieved_bio['documents'] and retrieved_bio['documents'][0]:
            for i in range(len(retrieved_bio['documents'][0])):
                distance = retrieved_bio['distances'][0][i]
                content = retrieved_bio['documents'][0][i]
                metadata = retrieved_bio["metadatas"][0][i]

                if distance <= threshold and not metadata.get("is_core", False):
                    general_docs.append(content)

        if general_docs:
            for doc in general_docs:
                bio_general_result += f"- {doc}\n"
        else:
            bio_general_result = ""

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": None,
            "tools_result": None,
            "bio_result": [bio_core_result, bio_general_result],
            "query": state["query"],
            "final_answer": None
        }

    def fusiontool_query_or_respond(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"]) + state["bio_result"][0]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + [ToolMessage(content=state["bio_result"][1], tool_call_id="temp")] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        openai_formatted_tools = [convert_to_openai_tool(tool) for tool in self.tool_list]

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)
            print('\n\n\n\n')

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)
            print('\n\n\n\n')

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)
        print('\n\n\n\n')
 
        if response.tool_calls:
            return {
                "variables": state["variables"],
                "system_prompt": state["system_prompt"],
                "branch_name": state["branch_name"],
                "messages": [response],
                "tools_result": None,
                "query": state["query"],
                "final_answer": None
            }

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "messages": None,
            "tools_result": None,
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": None
        }

    def fusiontool_check_for_tools(self, state: State):
        if state.get("messages"):
            return "tools"
        else:
            return "no_tool"

    def fusiontool_run_tools_and_pass_through_state(self, state: State):
        tools_result = self.tools.invoke(state["messages"])

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": tools_result,
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": None
        }

    def fusiontool_generate(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"]) + state["bio_result"][0]

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]
        
        if state["tools_result"]:
            trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + state["tools_result"] + [ToolMessage(content=state["bio_result"][1], tool_call_id="temp")] + [state["query"]])
        else:
            trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [ToolMessage(content=state["bio_result"][1], tool_call_id="temp")] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        pprint(openai_formatted_trimmed_messages)

        if self.formatter:
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)

        if state["tools_result"]:
            add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]
        else:
            add_messages = [state["query"]] + [response]

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "history": add_messages,
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": response
        }

    def fusiontool_extract_and_save_bio_memory(self, state:State):
        start = time.time()
        bio_extraction_prompt = self.config["BIO_EXTRACTION_PROMPT"]
        trimmed_messages = self.trimmer.invoke([SystemMessage(bio_extraction_prompt)] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),    
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", 40),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)
        
        if response:
            bio_list = parse_bio_with_importance(response.content)
            if bio_list:
                self.bio_metadata.save_or_update_bio(bio_list)
        end = time.time()

        print(f"extract_and_save_bio_memory 실행 시간: {end - start:.5f}초")

        return {
            "variables": state["variables"],
            "system_prompt": state["system_prompt"],
            "branch_name": state["branch_name"],
            "classifier_result": state["classifier_result"],
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "bio_result": state["bio_result"],
            "query": state["query"],
            "final_answer": None
        }

    
    # branch name: fusiontool_v2 (fusiontool + streaming + bio scheduler)

    def fusiontool_v2_check_thinking(self, state: State):
        if self.classifier_tokenizer == None or self.classifier_llm == None:
            print("에러: classifier_tokenizer 또는 classifier_llm 없음")
            sys.exit(1)

        id2label = {0: "Non-thinking", 1: "Thinking"}

        print(">>>query: " + state["query"].content)

        raw_tokens = self.classifier_tokenizer.encode(state["query"].content, add_special_tokens=False)

        if len(raw_tokens) > 510:
            print(f"Query가 너무 길어 512 토큰에 맞게 자르기 시행 ({len(raw_tokens)} -> 510)")
            raw_tokens = raw_tokens[:128] + raw_tokens[-382:]

        input_ids = [self.classifier_tokenizer.cls_token_id] + raw_tokens + [self.classifier_tokenizer.sep_token_id]

        inputs = {
            "input_ids": torch.tensor([input_ids]).to(self.classifier_llm.device),
            "attention_mask": torch.ones((1, len(input_ids))).to(self.classifier_llm.device)
        }

        with torch.no_grad():
            outputs = self.classifier_llm(**inputs)
    
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()

        label_name = id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")

        print(">>Label_name: " + label_name)

        return {
            "classifier_result": label_name,
        }
    
    def fusiontool_v2_retrieve_bio_memory(self, state: State):
        if self.kv_cache_snapshot is not None:
            start_time = time.time()
            self.llm.load_state(self.kv_cache_snapshot) # KV 캐시 스냅샷 로드
            self.kv_cache_snapshot = None # 로드 후 스냅샷 초기화
            print("Bio Memory 실행 후, 이전 대화 KV 캐시 스냅샷이 로드되었습니다.")
            end_time = time.time()
            print(f"KV 캐시 스냅샷 로드에 걸린 시간: {end_time - start_time:.2f} seconds")


        top_k = self.config["BIO_CONFIG"].get("top_k", 5)
        threshold = self.config["BIO_CONFIG"].get("retrieval_threshold_kor", 1.15)

        core_data = self.bio_metadata.get_bio_chroma_collection()._collection.get(
        where={"is_core": True}  
        )

        core_docs = core_data.get("documents", [])

        if core_docs:
            bio_core_result = self.config["CORE_BIO_EXPLANATION_PROMPT"]
            for doc in core_docs:
                bio_core_result += f"- {doc}\n"
        else:
            bio_core_result = ""

        parse_query_for_bio_result = parse_query_for_bio(state["query"].content)

        vector = self.bio_metadata.get_embedding_function().embed_query(parse_query_for_bio_result)
        retrieved_bio = self.bio_metadata.get_bio_chroma_collection()._collection.query(
            query_embeddings=[vector],
            n_results = top_k,
            include = ["documents", "metadatas", "distances"]
        )

        bio_general_result = self.config["BIO_EXPLANATION_PROMPT"]
        general_docs = []

        if retrieved_bio['documents'] and retrieved_bio['documents'][0]:
            for i in range(len(retrieved_bio['documents'][0])):
                distance = retrieved_bio['distances'][0][i]
                content = retrieved_bio['documents'][0][i]
                metadata = retrieved_bio["metadatas"][0][i]

                if distance <= threshold and not metadata.get("is_core", False):
                    print(f"Bio memory retrieved with distance {distance}: {content}")
                    general_docs.append(content)

        if general_docs:
            for doc in general_docs:
                bio_general_result += f"- {doc}\n"
        else:
            bio_general_result = ""

        return {
            "bio_result": [bio_core_result, bio_general_result],
        }

    def fusiontool_v2_query_or_respond(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"])

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        openai_formatted_tools = [convert_to_openai_tool(tool) for tool in self.tool_list]

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)
            print('\n\n\n\n')

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("top_k", 40),
                presence_penalty = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("presence_penalty", 0.3),
                repeat_penalty = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("repeat_penalty", 1.05),
                stream = False,
            )

            pprint(response_data)
            print('\n\n\n\n')

            text_output = response_data['choices'][0]['text'].strip()
        else:
            print("formatter = None")

            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                tools = openai_formatted_tools,
                max_tokens = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("top_k", 40),
                presence_penalty = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("presence_penalty", 0.3),
                repeat_penalty = self.config["CHAT_MODEL_CONFIG_TOOL_MODE"].get("repeat_penalty", 1.05),
                stream = False,
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)

        pprint(response)
        print('\n\n\n\n')
 
        if response.tool_calls:
            return {
                "messages": [response],
            }

        return

    def fusiontool_v2_check_for_tools(self, state: State):
        if state.get("messages"):
            return "tools"
        else:
            return "no_tool"

    def fusiontool_v2_run_tools_and_pass_through_state(self, state: State):
        tools_result = self.tools.invoke(state["messages"])

        return {
            "tools_result": tools_result,
        }

    def fusiontool_v2_generate(self, state: State, writer: StreamWriter):
        start_time_for_generate = time.time()
        thinking_mode_limitation_prompt = self.config["THINKING_MODE_LIMITATION_PROMPT"]

        if state["classifier_result"] == "Non-thinking":
            config_mode = "CHAT_MODEL_CONFIG"
            filled_system_prompt = state["system_prompt"].format(**state["variables"]) + state["bio_result"][0]
        else:
            config_mode = "CHAT_MODEL_CONFIG_THINKING_MODE"
            filled_system_prompt = state["system_prompt"].format(**state["variables"]) + thinking_mode_limitation_prompt + state["bio_result"][0]

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]
        
        if state["tools_result"]:
            trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + state["tools_result"] + [ToolMessage(content=state["bio_result"][1], tool_call_id="temp")] + [state["query"]])
        else:
            trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [ToolMessage(content=state["bio_result"][1], tool_call_id="temp")] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        pprint(openai_formatted_trimmed_messages)

        if self.formatter:
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            if state["classifier_result"] == "Non-thinking":
                full_prompt += '<think>\n\n</think>\n\n'

            print(full_prompt)

            response_generator = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config[config_mode].get("max_tokens", 16),
                temperature = self.config[config_mode].get("temperature", 0.8),
                top_p = self.config[config_mode].get("top_p", 0.95),
                min_p = self.config[config_mode].get("min_p", 0.05),
                stop = self.config[config_mode].get("stop", []),
                top_k = self.config[config_mode].get("top_k", 40),
                presence_penalty= self.config[config_mode].get("presence_penalty", 0.3),
                repeat_penalty= self.config[config_mode].get("repeat_penalty", 1.05),
                stream = True,    
            )

            pprint(response_generator)

            text_output = ""
            for chunk in response_generator:
                token = chunk['choices'][0]['text']
                text_output += token
                writer({"final_answer": token})
        else:
            print("formatter = None")

            response_generator = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config[config_mode].get("max_tokens", 16),
                temperature = self.config[config_mode].get("temperature", 0.8),
                top_p = self.config[config_mode].get("top_p", 0.95),
                min_p = self.config[config_mode].get("min_p", 0.05),
                stop = self.config[config_mode].get("stop", []),
                top_k = self.config[config_mode].get("top_k", 40),
                presence_penalty= self.config[config_mode].get("presence_penalty", 0.3),
                repeat_penalty= self.config[config_mode].get("repeat_penalty", 1.05),  
                stream = True,
            )

            pprint(response_generator)

            text_output = ""
            for chunk in response_generator:
                token = chunk['choices'][0]['message']['content']
                text_output += token
                writer({"final_answer": token})

        response = parse_llm_output(text_output)

        pprint(response)

        if state["tools_result"]:
            add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]
        else:
            add_messages = [state["query"]] + [response]

        # keep the queries in bio extraction buffer
        self.bio_extraction_buffer["queries"].append(state["query"].content)
        self.bio_extraction_buffer["token_count"] += self.get_num_tokens_from_query(state["query"])
        self.bio_extraction_buffer["query_count"] += 1
        print(f"현재 추가된 Bio 추출 버퍼 쿼리: {state['query'].content}")
        print(f"현재 추가된 쿼리 토큰 수: {self.get_num_tokens_from_query(state['query'])}")
        print(f"현재 Bio 추출 버퍼에 쌓인 토큰 수: {self.bio_extraction_buffer['token_count']}")
        print(f"현재 Bio 추출 버퍼에 쌓인 쿼리 수: {self.bio_extraction_buffer['query_count']}")

        end_time_for_generate = time.time()
        print(f"메인 답변 생성 실행 시간: {end_time_for_generate - start_time_for_generate:.5f}초")

        return {
            "history": add_messages,
        }
    
    def fusiontool_v2_check_for_bio_extraction(self, state: State):
        upcoming_id = state.get("upcoming_thread_id")
        buffer = self.bio_extraction_buffer
        threshold = self.config.get("BIO_CONFIG", {}).get("extraction_token_threshold", 512)
        
        if buffer.get("query_count", 0) == 0:
            return "skip_bio_extraction"

        # 대화창이 바뀌었을 때 (upcoming_thread_id가 현재 thread_id와 다르고, 버퍼에 쌓인 쿼리가 2개 이상일 때)
        is_new_thread = upcoming_id and upcoming_id != self.current_thread_id and buffer.get("query_count", 0) > 1
        
        # 버퍼에 쌓인 토큰 수가 임계치 이상일 경우
        is_threshold_met = buffer.get("token_count", 0) >= threshold

        if is_new_thread:
            self.current_thread_id = upcoming_id # ID 업데이트
            print("대화창이 변경되어 Bio 추출을 시작합니다.")
            return "extract_bio"
            
        if is_threshold_met:
            print("버퍼에 쌓인 토큰 수가 임계치를 초과하여 Bio 추출을 시작합니다.")
            return "extract_bio"

        self.current_thread_id = upcoming_id # ID 업데이트 (대화창이 바뀌지 않았더라도 ID는 업데이트)

        return "skip_bio_extraction"

    def fusiontool_v2_extract_and_save_bio_memory(self, state:State):
        start = time.time()

        self.kv_cache_snapshot = self.llm.save_state() # KV 캐시 스냅샷 저장
        print("Bio Memory 실행 전, 현재 대화 KV 캐시 스냅샷이 저장되었습니다.")

        bio_extraction_prompt = self.config["BIO_EXTRACTION_PROMPT_KOR"]
        buffer_messages = [HumanMessage(content=q) for q in self.bio_extraction_buffer["queries"]]

        trimmed_messages = self.trimmer.invoke([SystemMessage(bio_extraction_prompt)] + buffer_messages)

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

        if self.formatter:            
            full_prompt = self.formatter(
                messages = openai_formatted_trimmed_messages,
            ).prompt

            response_data = self.llm.create_completion(
                prompt = full_prompt,
                max_tokens = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("top_k", 40),
                presence_penalty = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("presence_penalty", 0.0),
                repeat_penalty = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("repeat_penalty", 1.0),
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['text'].strip()
        else:
            response_data = self.llm.create_chat_completion(
                messages = openai_formatted_trimmed_messages,
                max_tokens = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("max_tokens", 16),
                temperature = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("temperature", 0.8),
                top_p = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("top_p", 0.95),
                min_p = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("min_p", 0.05),
                stop = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("stop", []),
                top_k = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("top_k", 40),
                presence_penalty = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("presence_penalty", 0.0),
                repeat_penalty = self.config["CHAT_MODEL_CONFIG_BIO_EXTRACTION"].get("repeat_penalty", 1.0),  
            )

            pprint(response_data)

            text_output = response_data['choices'][0]['message']['content'].strip()
        
        response = parse_llm_output(text_output)
        
        if response:
            bio_list = parse_bio_with_importance(response.content)
            if bio_list:
                self.bio_metadata.save_or_update_bio(bio_list)
        end = time.time()

        print(f"extract_and_save_bio_memory 실행 시간: {end - start:.5f}초")
        self.bio_extraction_buffer= {
                "queries": [],
                "token_count": 0,
                "query_count": 0
        }

        return
    
    # 그래프 생성 함수

    def create_workflow(self):
        workflow = StateGraph(state_schema = State)

        # 노드 추가
        # branch name: default
        workflow.add_node("default_generate", self.default_generate)
        # branch name: tools
        workflow.add_node("tools_query_or_respond", self.tools_query_or_respond)
        workflow.add_node("tools_check_for_tools", self.tools_check_for_tools)
        workflow.add_node("tools_run_tools_and_pass_through_state", self.tools_run_tools_and_pass_through_state)
        workflow.add_node("tools_generate", self.tools_generate)
        # branch name: classifier
        workflow.add_node("classifier_check_thinking", self.classifier_check_thinking)
        workflow.add_node("classifier_query_or_respond", self.classifier_query_or_respond)
        workflow.add_node("classifier_check_for_tools", self.classifier_check_for_tools)
        workflow.add_node("classifier_run_tools_and_pass_through_state", self.classifier_run_tools_and_pass_through_state)
        workflow.add_node("classifier_generate", self.classifier_generate)
        # branch name: bio
        workflow.add_node("bio_generate", self.bio_generate)
        workflow.add_node("bio_retrieve_bio_memory", self.bio_retrieve_bio_memory)
        #workflow.add_node("bio_check_for_bio_extraction", self.bio_check_for_bio_extraction)
        workflow.add_node("bio_extract_and_save_bio_memory", self.bio_extract_and_save_bio_memory)
        # branch name: stream
        workflow.add_node("stream_check_thinking", self.stream_check_thinking)
        workflow.add_node("stream_query_or_respond", self.stream_query_or_respond)
        workflow.add_node("stream_check_for_tools", self.stream_check_for_tools)
        workflow.add_node("stream_run_tools_and_pass_through_state", self.stream_run_tools_and_pass_through_state)
        workflow.add_node("stream_generate", self.stream_generate)
        # branch name: fusion
        workflow.add_node("fusion_check_thinking", self.fusion_check_thinking)
        workflow.add_node("fusion_retrieve_bio_memory", self.fusion_retrieve_bio_memory)
        workflow.add_node("fusion_query_or_respond", self.fusion_query_or_respond)
        workflow.add_node("fusion_check_for_tools", self.fusion_check_for_tools)
        workflow.add_node("fusion_run_tools_and_pass_through_state", self.fusion_run_tools_and_pass_through_state)
        workflow.add_node("fusion_generate", self.fusion_generate)
        workflow.add_node("fusion_extract_and_save_bio_memory", self.fusion_extract_and_save_bio_memory)
        # branch name: fusiontool
        workflow.add_node("fusiontool_check_thinking", self.fusiontool_check_thinking)
        workflow.add_node("fusiontool_retrieve_bio_memory", self.fusiontool_retrieve_bio_memory)
        workflow.add_node("fusiontool_query_or_respond", self.fusiontool_query_or_respond)
        workflow.add_node("fusiontool_check_for_tools", self.fusiontool_check_for_tools)
        workflow.add_node("fusiontool_run_tools_and_pass_through_state", self.fusiontool_run_tools_and_pass_through_state)
        workflow.add_node("fusiontool_generate", self.fusiontool_generate)
        workflow.add_node("fusiontool_extract_and_save_bio_memory", self.fusiontool_extract_and_save_bio_memory)
        # branch name: fusiontool_v2
        workflow.add_node("fusiontool_v2_check_thinking", self.fusiontool_v2_check_thinking)
        workflow.add_node("fusiontool_v2_retrieve_bio_memory", self.fusiontool_v2_retrieve_bio_memory)
        workflow.add_node("fusiontool_v2_query_or_respond", self.fusiontool_v2_query_or_respond)
        workflow.add_node("fusiontool_v2_check_for_tools", self.fusiontool_v2_check_for_tools)
        workflow.add_node("fusiontool_v2_run_tools_and_pass_through_state", self.fusiontool_v2_run_tools_and_pass_through_state)
        workflow.add_node("fusiontool_v2_generate", self.fusiontool_v2_generate)
        workflow.add_node("fusiontool_v2_check_for_bio_extraction", self.fusiontool_v2_check_for_bio_extraction)
        workflow.add_node("fusiontool_v2_extract_and_save_bio_memory", self.fusiontool_v2_extract_and_save_bio_memory)


        # 노드 연결
        # 시작
        workflow.add_conditional_edges(START, self.router, {"default": "default_generate", "tools": "tools_query_or_respond", "classifier": "classifier_check_thinking", "bio": "bio_retrieve_bio_memory", "stream": "stream_check_thinking", "fusion": "fusion_check_thinking", "fusiontool": "fusiontool_check_thinking", "fusiontool_v2": "fusiontool_v2_check_thinking"})
        # branch name: default
        workflow.add_edge("default_generate", END)
        # branch name: tools
        workflow.add_conditional_edges("tools_query_or_respond", self.tools_check_for_tools, {"no_tool": END, "tools": "tools_run_tools_and_pass_through_state"})
        workflow.add_edge("tools_run_tools_and_pass_through_state", "tools_generate")
        workflow.add_edge("tools_generate", END)
        # branch name: classifier
        workflow.add_edge("classifier_check_thinking", "classifier_query_or_respond")
        workflow.add_conditional_edges("classifier_query_or_respond", self.classifier_check_for_tools, {"no_tool": END, "tools": "classifier_run_tools_and_pass_through_state"})
        workflow.add_edge("classifier_run_tools_and_pass_through_state", "classifier_generate")
        workflow.add_edge("classifier_generate", END)
        # branch name: bio
        workflow.add_edge("bio_retrieve_bio_memory", "bio_generate")
        workflow.add_conditional_edges("bio_generate", self.bio_check_for_bio_extraction, {"extract_bio": "bio_extract_and_save_bio_memory", "skip_bio_extraction": END})
        workflow.add_edge("bio_extract_and_save_bio_memory", END)
        # branch name: stream
        workflow.add_edge("stream_check_thinking", "stream_query_or_respond")
        workflow.add_conditional_edges("stream_query_or_respond", self.stream_check_for_tools, {"no_tool": END, "tools": "stream_run_tools_and_pass_through_state"})
        workflow.add_edge("stream_run_tools_and_pass_through_state", "stream_generate")
        workflow.add_edge("stream_generate", END)
        # branch name: fusion
        workflow.add_edge("fusion_check_thinking", "fusion_retrieve_bio_memory")
        workflow.add_edge("fusion_retrieve_bio_memory", "fusion_query_or_respond")
        workflow.add_conditional_edges("fusion_query_or_respond", self.fusion_check_for_tools, {"no_tool": "fusion_extract_and_save_bio_memory", "tools": "fusion_run_tools_and_pass_through_state"})
        workflow.add_edge("fusion_run_tools_and_pass_through_state", "fusion_generate")
        workflow.add_edge("fusion_generate", "fusion_extract_and_save_bio_memory")
        workflow.add_edge("fusion_extract_and_save_bio_memory", END)
        # branch name: fusiontool
        workflow.add_edge("fusiontool_check_thinking", "fusiontool_retrieve_bio_memory")
        workflow.add_edge("fusiontool_retrieve_bio_memory", "fusiontool_query_or_respond")
        workflow.add_conditional_edges("fusiontool_query_or_respond", self.fusiontool_check_for_tools, {"no_tool": "fusiontool_generate", "tools": "fusiontool_run_tools_and_pass_through_state"})
        workflow.add_edge("fusiontool_run_tools_and_pass_through_state", "fusiontool_generate")
        workflow.add_edge("fusiontool_generate", "fusiontool_extract_and_save_bio_memory")
        workflow.add_edge("fusiontool_extract_and_save_bio_memory", END)
        # branch name: fusiontool_v2
        workflow.add_edge("fusiontool_v2_check_thinking", "fusiontool_v2_retrieve_bio_memory")
        workflow.add_edge("fusiontool_v2_retrieve_bio_memory", "fusiontool_v2_query_or_respond")
        workflow.add_conditional_edges("fusiontool_v2_query_or_respond", self.fusiontool_v2_check_for_tools, {"no_tool": "fusiontool_v2_generate", "tools": "fusiontool_v2_run_tools_and_pass_through_state"})
        workflow.add_edge("fusiontool_v2_run_tools_and_pass_through_state", "fusiontool_v2_generate")
        workflow.add_conditional_edges("fusiontool_v2_generate", self.fusiontool_v2_check_for_bio_extraction, {"extract_bio": "fusiontool_v2_extract_and_save_bio_memory", "skip_bio_extraction": END})
        workflow.add_edge("fusiontool_v2_extract_and_save_bio_memory", END)


        # 메모리 추가
        memory = SqliteSaver(conn=sqlite3.connect(SQLITE_DB_FILE, check_same_thread = False))

        return workflow.compile(checkpointer = memory)