import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 

from src.db.chat_metadata import init_chat_metadata_db
from src.ui.app import create_simple_ui


if __name__ == "__main__":

    print("메타데이터 설정 시작...")
    init_chat_metadata_db()
    print("메타데이터 설정 끝")

    print("챗봇 UI 설정 시작...")
    interface = create_simple_ui()
    print("챗봇 UI 설정 끝")

    print("챗봇 시작...")
    interface.launch(share=True, inbrowser=True, show_error=True)
    print("챗봇 끝")