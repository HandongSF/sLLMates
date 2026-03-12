import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 

from src.db.chat_metadata import init_chat_metadata_db
from src.core.agent import ChatAgent
from src.core.app import create_simple_ui


if __name__ == "__main__":

    print("메타데이터 설정")
    init_chat_metadata_db()

    print("Agent 설정")
    agent = ChatAgent()

    print("챗봇 인터페이스 설정")
    interface = create_simple_ui(agent)

    print("챗봇 실행")
    interface.launch(share=True, inbrowser=True, show_error=True)

    print("프로그램 종료")