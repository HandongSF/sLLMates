import uuid
import re
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from langchain.schema import HumanMessage, AIMessage

from src.core.agent import ChatAgent
from src.db.chat_metadata import (
    save_chat_metadata, update_chat_metadata, get_chat_list,
    delete_chat, rename_chat, get_chat_name, generate_chat_name_from_message
)

pending_messages = {}


_BASE = os.path.dirname(__file__)

app = Flask(
    __name__,
    template_folder=os.path.join(_BASE, "templates"),
    static_folder=os.path.join(_BASE, "static"),
)

app.config["PROPAGATE_EXCEPTIONS"] = True

agent = None
bio_metadata = None


# ─── Helpers ────────────────────────────────────────────────────────────────

def remove_think(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def format_history_for_client(thread_data):
    """LangGraph state → [{role, content}] 형식으로 변환"""
    if not thread_data or "history" not in thread_data:
        return []
    messages = []
    for msg in thread_data.get("history", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    return messages


# ─── Page ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── Chat APIs ───────────────────────────────────────────────────────────────

@app.route("/api/chats", methods=["GET"])
def get_chats():
    """채팅 목록 반환"""
    chats = get_chat_list()  # [(name, id), ...]
    return jsonify([{"id": c[1], "name": c[0]} for c in chats])


@app.route("/api/chats", methods=["POST"])
def create_chat():
    """새 채팅 생성"""
    new_id = str(uuid.uuid4())
    chat_name = f"채팅 {datetime.now().strftime('%m/%d %H:%M')}"
    save_chat_metadata(new_id, chat_name)
    return jsonify({"id": new_id, "name": chat_name})


@app.route("/api/chats/<thread_id>", methods=["GET"])
def get_chat(thread_id):
    """특정 채팅방의 히스토리 반환"""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.app.get_state(config)
        history = format_history_for_client(state.values)
        name = get_chat_name(thread_id)
        return jsonify({"id": thread_id, "name": name, "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chats/<thread_id>", methods=["PATCH"])
def update_chat_name(thread_id):
    """채팅방 이름 변경"""
    data = request.get_json()
    new_name = (data.get("name") or "").strip()
    if not new_name:
        return jsonify({"error": "이름을 입력하세요"}), 400
    if rename_chat(thread_id, new_name):
        return jsonify({"id": thread_id, "name": new_name})
    return jsonify({"error": "변경 실패"}), 500


@app.route("/api/chats/<thread_id>", methods=["DELETE"])
def remove_chat(thread_id):
    """채팅방 삭제"""
    if delete_chat(thread_id):
        return jsonify({"ok": True})
    return jsonify({"error": "삭제 실패"}), 500


@app.route("/api/chats/<thread_id>/message", methods=["POST"])
def post_message(thread_id):
    """메시지를 등록하고 stream_id 반환"""
    data = request.get_json()
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "메시지를 입력하세요"}), 400
    
    stream_id = str(uuid.uuid4())
    pending_messages[stream_id] = {"thread_id": thread_id, "message": message}
    return jsonify({"stream_id": stream_id})


@app.route("/api/stream/<stream_id>", methods=["GET"])
def stream_response(stream_id):
    """EventSource용 GET 스트리밍"""
    if stream_id not in pending_messages:
        return jsonify({"error": "유효하지 않은 stream_id"}), 404
    
    item = pending_messages.pop(stream_id)
    thread_id = item["thread_id"]
    message = item["message"]

    # 첫 메시지면 자동 이름 설정
    new_name = None
    try:
        config_check = {"configurable": {"thread_id": thread_id}}
        state = agent.app.get_state(config_check)
        history = format_history_for_client(state.values)
        if len(history) == 0:
            auto_name = generate_chat_name_from_message(message)
            rename_chat(thread_id, auto_name)
            new_name = auto_name
    except Exception:
        pass


    def generate():
        if new_name:
            yield f"data: {json.dumps({'type': 'rename', 'name': new_name}, ensure_ascii=False)}\n\n"

        config = {"configurable": {"thread_id": thread_id}}
        input_messages = HumanMessage(content=message)
        display_text = ""
        is_intercepted = False

        try:
            for mode, step in agent.app.stream(
                {
                    "variables": agent.config.get("VARIABLES", {}),
                    "system_prompt": agent.config.get("SYSTEM_PROMPT", ""),
                    "branch_name": "fusiontool_v2", # switch the branch to try other workflows
                    "upcoming_thread_id": thread_id,
                    "classifier_result": None,
                    "messages": None,
                    "tools_result": None,
                    "bio_result": None,
                    "query": input_messages,
                    "final_answer": None,
                },
                config=config,
                stream_mode=["values", "custom"],
            ):
                if mode == "values":
                    raw_text = step["final_answer"]

                    if hasattr(raw_text, "content"):
                        raw_text = raw_text.content
                        display_text += remove_think(raw_text)
                        chunk = {"type": "token", "content": display_text}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                if mode == "custom":
                    if "final_answer" in step and step["final_answer"] is not None:
                        raw_text = step["final_answer"]

                        if "<think>" in raw_text:
                            is_intercepted = True
                            chunk = {"type": "status", "content": "Thinking 모드 사용중..."}
                        elif "</think>" in raw_text:
                            is_intercepted = False
                            continue
                        elif "<tool_call>" in raw_text:
                            is_intercepted = True
                            chunk = {"type": "status", "content": "도구를 사용하고 있습니다..."}
                        elif "</tool_call>" in raw_text:
                            is_intercepted = False
                            continue
                        elif not is_intercepted:
                            chunk = {"type": "token", "content": raw_text}
                        else:
                            continue

                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            update_chat_metadata(thread_id)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            print(f"스트리밍 오류: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


# ─── Bio APIs ────────────────────────────────────────────────────────────────

@app.route("/api/bios", methods=["GET"])
def get_bios():
    bios = bio_metadata.get_all_bios()
    return jsonify(bios or [])


@app.route("/api/bios", methods=["POST"])
def create_bio():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    importance = int(data.get("importance", 5))

    if not text:
        return jsonify({"error": "텍스트를 입력하세요"}), 400
    if not (1 <= importance <= 10):
        return jsonify({"error": "중요도는 1~10 사이여야 합니다"}), 400

    try:
        bio_id = bio_metadata.add_bio(text, importance)
        return jsonify({"id": bio_id, "document": text, "importance": importance})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bios/<bio_id>", methods=["PATCH"])
def update_bio(bio_id):
    data = request.get_json()
    text = (data.get("text") or "").strip()
    importance = data.get("importance")

    if not text:
        return jsonify({"error": "텍스트를 입력하세요"}), 400

    try:
        importance_val = int(importance) if importance is not None else None
        bio_metadata.update_bio(bio_id, text=text, importance=importance_val)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bios/<bio_id>", methods=["DELETE"])
def delete_bio_route(bio_id):
    try:
        bio_metadata.delete_bio(bio_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def create_simple_ui(chat_agent: ChatAgent):
    global agent, bio_metadata
    agent = chat_agent
    bio_metadata = chat_agent.bio_metadata
    return app


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)