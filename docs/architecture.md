# 프로젝트 아키텍쳐 구조

## 디렉토리 구조

프로젝트는 기능별로 명확하게 분리된 구조를 가짐

```
SLLMATES/
├── data/                 # 데이터(문서, DB) 저장소
├── docs/                 # 프로젝트 상세 문서 저장소
├── models/               # LLM, 임베딩 모델 저장소
├── src/                  # 전체 소스 코드
│   ├── config/           # 설정별 config 파일 저장소
│   ├── core/             # 모델 로딩, 에이전트, 도구 정의
│   ├── db/               # 채팅 DB 관리 로직
│   ├── config.py         # 경로 설정, 사용할 config 폴더 내부 파일 설정
├── .gitignore            # 깃허브에 올리지 않을 프로젝트 파일 설정
├── CODE_OF_CONDUCT.txt   # 프로젝트 라이센스 관련 설명
├── LICENSE               # 프로젝트 라이센스 관련 설명
├── main.py               # 애플리케이션 시작점
├── NOTICE.txt            # 프로젝트 라이센스 관련 설명
└── README.md             # 프로젝트 기본 설명
```

---