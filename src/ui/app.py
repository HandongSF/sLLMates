import gradio as gr
import uuid
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage

from src.core.agent import LangChainAgent
from src.config import SYSTEM_PROMPT, VARIABLES
from src.db.metadata import (
    save_chat_metadata, update_chat_metadata, get_chat_list,
    delete_chat, rename_chat, get_chat_name, generate_chat_name_from_message
)


agent = LangChainAgent()


# UI Helper Functions
def create_chatbot_response(message, history, thread_id):
    if thread_id and len(history) == 1 and "새 채팅이 시작되었습니다" in (history[0][1] or ""):
        auto_name = generate_chat_name_from_message(message)
        rename_chat(thread_id, auto_name)

    if not thread_id:
        history.append([message, "⚠️ '새 채팅'을 눌러 대화를 시작해주세요."])
        yield history, ""
        return

    if not message.strip():
        yield history, ""
        return

    if message.lower().strip() in ["exit", "q", "끝"]:
        history.append([message, "대화를 종료합니다."])
        yield history, ""
        return

    # 초기 표시
    history.append([message, "💭 생각 중..."])
    yield history, ""

    config = {"configurable": {"thread_id": thread_id}}

    try:
        input_messages = HumanMessage(content=message)
        partial_response = ""

        # LangChain streaming
        for step in agent.app.stream(
            {
                "variables": VARIABLES,
                "system_prompt": SYSTEM_PROMPT,
                "messages": None,
                "tools_result": None,
                "query": input_messages,
                "final_answer": None
            },
            config=config,
            stream_mode="values",
        ):
            if "final_answer" in step and step["final_answer"]:
                text_piece = (
                    step["final_answer"].content
                    if hasattr(step["final_answer"], "content")
                    else str(step["final_answer"])
                )

                # 🔤 한 글자씩 표시
                for ch in text_piece:
                    partial_response += ch
                    history[-1][1] = partial_response
                    yield history, ""

        # 메타데이터 업데이트
        update_chat_metadata(thread_id)

    except Exception as e:
        print(f"응답 생성 오류: {e}")
        history[-1][1] = f"❌ 오류가 발생했습니다: {str(e)}"
        yield history, ""

#LangGraph state를 Gradio Chatbot 형식으로 변환
def format_history_for_chatbot(thread_data):
    if not thread_data or 'history' not in thread_data:
        return []
    
    history = thread_data.get('history', [])
    chatbot_history = []
    
    for msg in history:
        if isinstance(msg, HumanMessage):
            chatbot_history.append([msg.content, None])
        elif isinstance(msg, AIMessage):
            if chatbot_history and chatbot_history[-1][1] is None:
                chatbot_history[-1][1] = msg.content
            else:
                chatbot_history.append([None, msg.content])
    
    return chatbot_history


# Gradio UI 
def create_simple_ui():
    css = """
    .gradio-container { 
        min-height: 80vh;
        width: 80vh;
        margin: auto !important; 
        
    }
    #sidebar {
        background-color: #f8f9fa;
        border-right: 2px solid #e0e0e0;
        padding: 20px 15px;
        min-height: 100vh;
    }
    #chat-history-dropdown ul.options {
        position: static !important; /* 원래 위치에 고정 */
        max-height: none !important; /* 최대 높이 제한 해제 */
    }
    #chatbox {
        height: 80vh !important;
    }
    """
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 💬 RAG 챗봇")

        thread_id_state = gr.State(None)
        
        with gr.Row():
            with gr.Sidebar(position="left"):
                gr.Markdown("## 📑 채팅방 설정")
                with gr.Row():
                    new_chat_btn = gr.Button("📝새 채팅", variant="primary", size="md")
                    delete_btn = gr.Button("🗑️ 채팅방 삭제", variant="stop", size="md")

                gr.Markdown("---")
                gr.Markdown("### 채팅방 이름 바꾸기")
                with gr.Row():
                    rename_input = gr.Textbox(
                        placeholder="새로운 채팅방 이름",
                        show_label=False,
                        scale=3,
                        container=False
                    )
                    rename_btn = gr.Button("✏️ 채팅방 이름 수정", size="md", scale=3)
                
                gr.Markdown("---")
                gr.Markdown("### 현재 채팅방")
                current_chat_info = gr.Markdown("없음", elem_classes=["medium-text"])

                gr.Markdown("---")
                gr.Markdown("### 채팅방 목록")
                chat_dropdown = gr.Dropdown(
                    label="채팅방 목록",
                    choices=[],
                    interactive=True,
                    container=False,
                    elem_id="chat-history-dropdown"
                )

            with gr.Column(scale=10):
                chatbot = gr.Chatbot(
                    show_label=False,
                    avatar_images=None, 
                    show_copy_button=True,
                    elem_id="chatbox"
                )

                with gr.Row():
                    msg = gr.Textbox(
                        show_label=False, 
                        placeholder="메시지를 입력하세요...",
                        container=False, 
                        scale=10
                    )
                    submit = gr.Button("📤 제출", variant="primary", scale=2)

        # ==================== 이벤트 핸들러 ====================
        
        
        def start_new_chat():
            new_id = str(uuid.uuid4())
            chat_name = f"채팅[{datetime.now().strftime('%m/%d %H:%M:%S]')}"
            
            save_chat_metadata(new_id, chat_name)

            updated_choices =get_chat_list() 
            
            return (
                new_id,
                [[None, f"✨ 새 채팅이 시작되었습니다."]],
                "",
                gr.update(choices=updated_choices, value=new_id),
                f"📍 {chat_name}"
            )
        
        def load_chat_history(selected_thread_id):
            if not selected_thread_id:
                return [], "없음", None
            
            try:
                config = {"configurable": {"thread_id": selected_thread_id}}
                state = agent.app.get_state(config)
                history = format_history_for_chatbot(state.values)
                
                chat_name = get_chat_name(selected_thread_id)
                info_text = f"📍 {chat_name}"
                
                return history, info_text, selected_thread_id
            except Exception as e:
                print(f"채팅 로드 오류: {e}")
                return [], f"❌ 로드 실패", None

        def refresh_list(current_id):
            choices = get_chat_list()
            value = current_id if current_id and any(c[1] == current_id for c in choices) else None
            return gr.update(choices=choices, value=value)

        def rename_current(thread_id, new_name):
            if not thread_id:
                return gr.update(), "", "⚠️ 선택된 채팅 없음"
            
            if not new_name.strip():
                return gr.update(), "", "⚠️ 이름을 입력하세요"
            
            if rename_chat(thread_id, new_name.strip()):
                return refresh_list(thread_id), "", f"📍 {new_name.strip()}"
            return gr.update(), "", "❌ 변경 실패"

        def delete_current(thread_id):
            if not thread_id:
                return gr.update(), [], "없음", None
            
            if delete_chat(thread_id):
                choices = get_chat_list()
                return gr.update(choices=choices, value=None), [], "없음", None
            return gr.update(), gr.update(), gr.update(), thread_id

        # 이벤트 바인딩
        new_chat_btn.click(
            start_new_chat,
            outputs=[thread_id_state, chatbot, msg, chat_dropdown, current_chat_info]
        )

        chat_dropdown.change(
            load_chat_history,
            inputs=[chat_dropdown],
            outputs=[chatbot, current_chat_info, thread_id_state]
        )

        rename_btn.click(
            rename_current,
            inputs=[thread_id_state, rename_input],
            outputs=[chat_dropdown, rename_input, current_chat_info]
        )

        delete_btn.click(
            delete_current,
            inputs=[thread_id_state],
            outputs=[chat_dropdown, chatbot, current_chat_info, thread_id_state]
        )

        # 메시지 전송 (streaming 적용)
        submit.click(
            fn=create_chatbot_response,           # 기존 send_message_with_update 대신 직접 사용
            inputs=[msg, chatbot, thread_id_state],
            outputs=[chatbot, msg]
        )

        msg.submit(
            fn=create_chatbot_response,
            inputs=[msg, chatbot, thread_id_state],
            outputs=[chatbot, msg]
        )

        # 초기 로드
        interface.load(
            lambda: gr.update(choices=get_chat_list()),
            outputs=[chat_dropdown]
        )

        interface.load(
            start_new_chat,
            outputs=[thread_id_state, chatbot, msg, chat_dropdown, current_chat_info]
        )

    return interface