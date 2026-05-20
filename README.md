# sLLMates

[![Static Badge](https://img.shields.io/badge/license-apache_2.0-blue)](https://opensource.org/license/apache-2.0)

로컬 LLM을 사용하는 에이전트 챗봇 시스템

## 주요 기능

* 에이전트 시스템: LLM이 사용자의 질의를 해결하기 위해 능동적으로 문제 해결 방법을 구상하고 도구 사용을 판단.
* LLM 답변 최적화: 사용자 질의의 난이도에 따라 LLM의 Thinking Mode 사용을 제어하여 답변 속도를 최적화.
* 사용자 맞춤 메모리: 사용자 질의에 포함된 맞춤 정보를 추출 및 저장하여 추후 대화에서 답변 개선에 사용.
* 대화 기록 관리: 사용자와의 대화 기록을 데이터베이스에 저장.
* 편리한 채팅 관리: 사용자의 채팅 기록을 채팅방 단위로 관리할 수 있도록 UI에서 기능 제공.

## 현재 사용 가능 도구

* RAG(Retrieval-Augmented Generation)

## Quick start

### Requirements

* Python 3.9 버전 이상 필요
* GPU(NVIDIA) 사용 시 CUDA Toolkit 설치 필요

### Clone Project

```bash
# Clone project code
git clone https://github.com/HandongSF/sLLMates.git

# Move into project folder
cd sLLMates
```

### Set Enviroment

```bash
# Make conda enviroment and install packages
conda env create -f environment.yml

# Activate conda enviroment
conda activate sLLMates

# Install Llama-cpp-python with Using CUDA option
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

### Download Model

1. 루트 폴더의 models/ 내부에 허깅페이스 등에서 다운로드한 로컬 모델의 gguf 파일과 기타 모델 파일을 추가(만약 임베딩 모델이 필요할 경우 임베딩 모델은 양자화하지 않은 상태로 넣을 것)
2. 그리고 루트 폴더의 src/configs/ 내부에 자신의 모델 세팅에 맞는 세팅 파일 제작 후 추가  

### Start Project

```bash
# Start project
python main.py

# 만약 GPU를 2개 이상 사용한다면 세팅 파일의 tensor_split 옵션 조정 후 아래 명령어 사용
# CUDA_VISIBLE_DEVICES=0,1,... python main.py
```

## Other documentation

시스템 구조, 내부 원리 등은 루트 폴더의 docs 내부 문서를 참조