import sys
import sqlite3
import time
from pprint import pprint
from datetime import datetime
import importlib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Sequence, Dict, List, Optional, Tuple, Annotated
from typing_extensions import Annotated, TypedDict
from llama_cpp import Llama
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
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from langchain_community.chat_models import ChatLlamaCpp
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import SELECTED_CONFIG_FILE, SQLITE_DB_FILE
from src.db.vector_store import ChromaDBManager
from src.db.bio_metadata import BioMetadata
from src.core.parsers import parse_llm_output, convert_messages_to_llama3_messages, parse_bio_with_importance


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

    messages: Optional[List[BaseMessage]]
    """tool 호출이 포함된 임시 메시지 (tool 실행용)"""

    tools_result: Optional[List[ToolMessage]]
    """tool 실행 결과 메시지"""

    bio_result: Tuple[str, str]
    """bio memory 검색 결과를 시스템 컨텍스트로 변환한 문자열"""

    query: HumanMessage
    """현재 사용자 입력"""

    final_answer: Optional[AIMessage]
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
            "branch_name": label_name,
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

            if state["branch_name"] == "Non-thinking":
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

            if state["branch_name"] == "Non-thinking":
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
            "messages": state["messages"],
            "tools_result": state["tools_result"],
            "query": state["query"],
            "final_answer": response
        }

    # branch name: 
    # branch name: bio

    def bio_retrieve_bio_memory(self, state: State):
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
            "bio_result": [bio_core_result, bio_general_result]
        }

    def bio_generate(self, state: State):
        filled_system_prompt = state["system_prompt"].format(**state["variables"]) + state["bio_result"][0]

        conversation_messages = [
            message
            for message in state["history"]
            if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
        ]

        trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [state["bio_result"][1]] + [state["query"]])

        openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

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

    def bio_extract_and_save_bio_memory(self, state:State):
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

        return 

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


    # def query_or_respond(self, state: State):
    #     filled_system_prompt = TOOL_PROMPT.format(**state["variables"])
    #     trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + [state["query"]])

    #     openai_formatted_trimmed_messages = convert_to_openai_messages(trimmed_messages)

    #     openai_formatted_tools = [convert_to_openai_tool(tool) for tool in TOOL_LIST]

    #     full_prompt = self.formatter.format(
    #         messages = openai_formatted_trimmed_messages,
    #         custom_tools = openai_formatted_tools,
    #         tools_in_user_message = False,
    #         date_string = datetime.now().strftime("%d %b %Y"),
    #         add_generation_prompt = True,
    #     )

    #     response_data = self.llm.create_completion(
    #         prompt = full_prompt,
    #         max_tokens = self.config["CHAT_MODEL_CONFIG"].get("max_tokens", ),
    #         temperature = self.config["CHAT_MODEL_CONFIG"].get("temperature", ),
    #         top_p = self.config["CHAT_MODEL_CONFIG"].get("top_p", ),
    #         min_p = self.config["CHAT_MODEL_CONFIG"].get("min_p", ),
    #         stop = self.config["CHAT_MODEL_CONFIG"].get("stop", ),
    #         top_k = self.config["CHAT_MODEL_CONFIG"].get("top_k", ),    
    #     )
        
    #     print("모델 답변: query_or_respond: " + repr(response_data))
    #     response = parse_llm_output(response_data)
    #     print("모델 답변 가공: query_or_respond: " + repr(response))
 
    #     if response.tool_calls:
    #         return {
    #             "variables": state["variables"],
    #             "system_prompt": state["system_prompt"],
    #             "messages": [response],
    #             "tools_result": None,
    #             "query": state["query"],
    #             "final_answer": None
    #         }

    #     return {
    #         "variables": state["variables"],
    #         "system_prompt": state["system_prompt"],
    #         "messages": None,
    #         "tools_result": None,
    #         "query": state["query"],
    #         "final_answer": None
    #     }

    # def check_for_tools(self, state: State):
    #     if state.get("messages"):
    #         return "tools"
    #     else:
    #         return "no_tool"

    # def run_tools_and_pass_through_state(self, state: State):
    #     tools_result = self.tools.invoke(state["messages"])

    #     return {
    #         "variables": state["variables"],
    #         "system_prompt": state["system_prompt"],
    #         "messages": state["messages"],
    #         "tools_result": tools_result,
    #         "query": state["query"],
    #         "final_answer": None
    #     }

    # def generate(self, state: State):
    #     filled_system_prompt = state["system_prompt"].format(**state["variables"]) + state["bio_result"]

    #     conversation_messages = [
    #         message
    #         for message in state["history"]
    #         if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
    #     ]

    #     if state["tools_result"]:
    #         trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)]+ conversation_messages + state["tools_result"] + [state["query"]])
    #         print("tools_result 존재")
    #     else:
    #         trimmed_messages = self.trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + [state["query"]])
    #         print("tools_result 없음")

    #     if USING_LLAMA:
    #         response_data = self.llm.create_completion(
    #             prompt = convert_messages_to_text_format_llama3(trimmed_messages),
    #             max_tokens = LLMConfig.max_tokens,
    #             temperature = LLMConfig.temperature,
    #             top_p = LLMConfig.top_p,
    #             min_p = LLMConfig.model_kwargs["min_p"],
    #             stop = LLMConfig.stop,
    #             top_k = LLMConfig.top_k,      
    #         )
    #         print("모델 답변: generate: " + repr(response_data))
    #         response = parse_llm_output(response_data)
    #         print("모델 답변 가공: generate: " + repr(response))
    #     else:
    #         response = self.llm.invoke(trimmed_messages)

    #     if state["tools_result"]:
    #         add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]
    #     else:
    #         add_messages = [state["query"]] + [response]

    #     return {
    #         "variables": state["variables"],
    #         "system_prompt": state["system_prompt"],
    #         "history": add_messages,
    #         "messages": state["messages"],
    #         "tools_result": state["tools_result"],
    #         "query": state["query"],
    #         "final_answer": response
    #     }
    
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
        # branch name: 
        # branch name: bio
        workflow.add_node("bio_generate", self.bio_generate)
        workflow.add_node("bio_retrieve_bio_memory", self.bio_retrieve_bio_memory)
        workflow.add_node("bio_extract_and_save_bio_memory", self.bio_extract_and_save_bio_memory)
        
        # workflow.add_node("retrieve_bio_memory", retrieve_bio_memory)
        # workflow.add_node("query_or_respond", query_or_respond)
        # workflow.add_node("run_tools_and_pass_through_state", run_tools_and_pass_through_state)
        # workflow.add_node("generate", self.generate)
        # workflow.add_node("extract_and_save_bio_memory", self.extract_and_save_bio_memory)

        # 노드 연결
        # 시작
        workflow.add_conditional_edges(START, self.router, {"default": "default_generate", "tools": "tools_query_or_respond", "classifier": "classifier_check_thinking", "bio": "bio_retrieve_bio_memory"})
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
        
        # branch name: tools
        # workflow.add_conditional_edges("tools_query_or_respond", self.tools_check_for_tools, {"no_tool": END, "tools": "tools_run_tools_and_pass_through_state"})
        # workflow.add_edge("tools_run_tools_and_pass_through_state", "tools_generate")
        # workflow.add_edge("tools_generate", END)
        
        # branch name: bio
        workflow.add_edge("bio_retrieve_bio_memory", "bio_generate")
        workflow.add_edge("bio_generate", "bio_extract_and_save_bio_memory")
        workflow.add_edge("bio_extract_and_save_bio_memory", END)
        # workflow.add_edge("retrieve_bio_memory", "query_or_respond")
        # workflow.add_conditional_edges("query_or_respond", self.check_for_tools, {"no_tool": "generate", "tools": "run_tools_and_pass_through_state"})
        # workflow.add_edge("run_tools_and_pass_through_state", "generate")
        # workflow.add_edge("generate", "extract_and_save_bio_memory")
        # workflow.add_edge("extract_and_save_bio_memory", END)

        # 메모리 추가
        memory = SqliteSaver(conn=sqlite3.connect(SQLITE_DB_FILE, check_same_thread = False))

        return workflow.compile(checkpointer = memory)