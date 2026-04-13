from src.config import MODELS_DIR



CONFIG = {

    # 채팅 모델 설정

    "CHAT_MODEL_CONFIG": {
        # class: Llama
        "model_path": str(MODELS_DIR / "Qwen3-14B-Q4_K_M.gguf"),
        # Model Params
        "n_gpu_layers": -1,
        "main_gpu": 0,
        "tensor_split": [0.3, 0.7],
        "use_mmap": True,
        "use_mlock": False,
        # Context Params
        "n_ctx": 4096,
        "n_batch": 512,
        "flash_attn": True,
        # Misc
        "verbose": True,
        
        # function: create_completion
        "max_tokens": -1,
        "temperature": 0.6,
        "top_p": 0.95,
        "min_p": 0,
        "stop": ["<|endoftext|>", "<|im_end|>"],
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "top_k": 20,
    },

    # 임베딩 모델 설정

    "EMBEDDING_MODEL_CONFIG": {
        "model_name": str(MODELS_DIR / "Qwen3-Embedding-0.6B"),
        "model_kwargs": {'device': 'cuda:0'},
        "encode_kwargs": {'normalize_embeddings': True},
    },

    # 트리머 설정

    "TRIMMER_CONFIG": {
        "max_tokens": 2000,
        "strategy": "last",
        "include_system": True,
        "allow_partial": False,
        "start_on": "human",
    },

    # 분류 모델 설정

    "CLASSIFIER_MODEL_CONFIG": {
        "model_path": str(MODELS_DIR / "finetuned_classifier_model_new_non"),
    },

    # RAG 설정

    "RAG_CONFIG": {
        "chunk_size": 200,
        "chunk_overlap": 50,
        "batch_size": 16,
        "retrieval_k": 5,
    },

    # Bio 설정

    "BIO_CONFIG": {
        "retrieval_threshold": float(1.0),
        "retrieval_threshold_kor": float(1.15),
        "top_k": 5,
        "extraction_token_threshold": 512,
    },

    # 시스템 프롬프트 및 변수 설정

    "VARIABLES": {
        "language": "Korean",
    },

    "SYSTEM_PROMPT": """
    You are a helpful assistant. The response language is {language}. Please provide short, concise, and clear answer to the user's question.
    """,

    "BIO_EXTRACTION_PROMPT": """
# Role: Selective Memory Architect (Memory Tagger)

Your task is to extract atomic facts from the conversation and assign an Importance Score (1-5).

# Importance Rubric (Strict)
1: Noise (Greetings, simple reactions like "okay", "wow")
2: Ephemeral (Current mood, immediate context, temporary states)
3: Preferences (Likes/dislikes, hobbies, habits, names of friends)
4: Life Pillars (Career, studies, long-term goals, social status)
5: Core Bio (Name, Age, Job, Nationality, MBTI - Mark is_core: true)

# Operational Rules
- **ATOMICITY**: Extract each distinct fact as a separate <bio> block.
- **SPO FORMAT**: Use "User Predicate Object" (e.g., "User majors in AI Engineering").
- **SCORE LIMIT**: Use ONLY integers 1, 2, 3, 4, 5.

# Output Format
<think>
1. Fact Extraction: [List candidate facts]
2. Scoring: [Fact] -> [Why it's useful/persistent] -> [Score]
3. Core Check: [Is it Name/Age/Job/Nationality/MBTI?]
</think>

<bio>
content: [SPO Sentence]
importance: [1-5]
is_core: [true/false]
</bio>
    
[Repeat <bio> blocks for every identified fact in the conversation.]

Below are the conversation sentences made by the user:
""",

    'BIO_EXTRACTION_PROMPT_V2': """
# Role: Selective Memory Architect (Memory Tagger)

Extract useful user facts from the conversation and assign an importance score (1-5).

# Importance Rubric
1: Noise (greetings, filler)
2: Temporary (current mood, short-term context)
3: Preferences (food, hobbies, habits)
4: Stable attributes (education, job, long-term roles)
5: Core identity (name, age, job, nationality, education)

# Rules
- Extract atomic but non-redundant facts
- Prefer concise natural sentences (not rigid SPO)
- Only store information useful for future interactions
- Avoid duplicates or overly similar facts

# Output Format
<bio>
content: ...
importance: 1-5
is_core: true/false
</bio>

Return ONLY <bio> blocks. No explanation.

User conversation:
    """,

    'BIO_EXTRACTION_PROMPT_V3': """

    """,

    "BIO_EXTRACTION_PROMPT_KOR": """
# Role: 개인 맞춤형 메모리 설계자 (Memory Tagger)

당신의 임무는 사용자(User)와의 대화에서 원자 단위의 사실(Atomic Facts)을 추출하고, 중요도 점수(1-5)를 부여하는 것입니다.

# 중요도 산정 기준 (Strict)
1점: 노이즈 (인사, 단순 리액션, "응", "와" 같은 추임새)
2점: 휘발성 정보 (현재 기분, "지금 ~하고 있어"와 같은 일시적 상태, 단순 정황)
3점: 개인적 취향 (좋아함/싫어함, 취미, 습관, 지인/가족 이름)
4점: 삶의 기둥 (직업, 전공, 거주지, 장기적인 목표, 핵심 가치관)
5점: 핵심 정체성 (이름, 나이, 국적, 성별, MBTI - 반드시 is_core: true로 표시)

# 실행 규칙
- **원자성(Atomicity)**: 하나의 <bio> 태그에는 오직 하나의 사실만 담으세요.
- **문장 형식**: "사용자는 대상을 서술함" 형태의 간결한 한국어 평서문을 사용하세요. (예: "사용자는 AI 공학을 전공함")
- **점수 제한**: 반드시 정수 1, 2, 3, 4, 5만 사용하세요. 사소한 대화는 엄격하게 1점을 부여하세요.

# 출력 형식
<think>
1. 정보 추출: [후보 사실들 나열]
2. 점수 산정: [사실] -> [이유/지속성 판단] -> [점수]
3. 핵심 정보 체크: [이름/나이/직업/국적/MBTI 여부 확인]
</think>

<bio>
content: [한국어 평서문]
importance: [1-5]
is_core: [true/false]
</bio>

아래는 사용자의 대화 문장들입니다:
    """,

    "BIO_EXPLANATION_PROMPT": """
### Background Context
Below are pieces of your internal knowledge about the user. Use them as your "implicit background knowledge" to inform your suggestions naturally, without ever explicitly mentioning or citing them:
""",

    "BIO_EXPLANATION_PROMPT_KOR": """
아래의 정보는 당신이 이미 알고 있는 사용자의 배경 지식이니, 대화 시 절대로 '기억'이나 '취향'을 근거로 언급하지 말고 오랜 친구로서의 직관인 것처럼 자연스럽게 답변에 반영하세요:
""",

    "CORE_BIO_EXPLANATION_PROMPT": """
### User Profile
Keep this core profile in mind to stay consistent with the user's background:
""",

    "TOOL_PROMPT": """
    You are Llama3.1, a large language model trained by Meta, based on the Llama architecture.
    You are chatting with the user via the Chating app.

    Never use emojis unless explicitly asked to.

    Your task is to evaluate ONLY the current user query and decide whether a tool call is needed.

    Strict rules for tool usage:
    1. Call a tool ONLY when the user is clearly requesting specific factual information 
       that requires external or up-to-date data that you cannot reliably answer from your internal knowledge.

    2. Do NOT call a tool for:
       - greetings or small talk
       - conceptual explanations or general knowledge
       - subjective questions, opinions, or reasoning tasks
       - ambiguous or underspecified queries
       - anything you can answer confidently without external data

    3. You must be at least 90% certain that the user expects information that requires a tool
       before performing a tool call.

    Output rules:
    - If a tool call IS needed:
        Output ONLY the valid tool-call JSON with no additional text.
    - If a tool call is NOT needed:
        Return in string the reason why you did not call tool in current case.

    The response language is {language}.

    """,

    # 커스텀 챗 핸들러 사용 설정

    "USE_CUSTOM_CHAT_HANDLER": True,

    "FORMATTER_CONFIG": {
        "eos_token": "<|im_end|>",
        "bos_token": "<|endoftext|>",
    },

    "CUSTOM_CHAT_TEMPLATE_VARIABLES": {
        ""
    },

    "CUSTOM_CHAT_TEMPLATE": r"""
    {%- if tools %} 
        {{- '<|im_start|>system\n' }} 
        {%- if messages[0].role == 'system' %} 
            {{- messages[0].content + '\n\n' }} 
        {%- endif %} 
        {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }} 
        {%- for tool in tools %} 
            {{- "\n" }} 
            {{- tool | tojson }} 
        {%- endfor %} 
        {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }} 
    {%- else %} 
        {%- if messages[0].role == 'system' %} 
            {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }} 
        {%- endif %} 
    {%- endif %} 
    
    {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %} 
    
    {%- for index in range(ns.last_query_index, -1, -1) %} 
        {%- set message = messages[index] %} 
        {%- if ns.multi_step_tool and message.role == "user" and not('<tool_response>' in message.content and '</tool_response>' in message.content) %} 
            {%- set ns.multi_step_tool = false %} 
            {%- set ns.last_query_index = index %} 
        {%- endif %} 
    {%- endfor %} 
    {%- for message in messages %} 
        {%- if (message.role == "user") or (message.role == "system" and not loop.first) %} 
            {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }} 
        {%- elif message.role == "assistant" %} 
            {%- set content = message.content %} 
            {%- set reasoning_content = '' %} 
            {%- if message.reasoning_content is defined and message.reasoning_content is not none %} 
                {%- set reasoning_content = message.reasoning_content %} 
            {%- else %} 
                {%- if '</think>' in message.content %} 
                    {%- set content = message.content.split('</think>')[-1].lstrip('\n') %} 
                    {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %} 
                {%- endif %} 
            {%- endif %} 
            {%- if loop.index0 > ns.last_query_index %} 
                {%- if loop.last or (not loop.last and reasoning_content) %} 
                    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }} 
                {%- else %} 
                    {{- '<|im_start|>' + message.role + '\n' + content }} 
                {%- endif %} 
            {%- else %} 
                {{- '<|im_start|>' + message.role + '\n' + content }} 
            {%- endif %} 

            {%- if message.tool_calls %} 
                {%- for tool_call in message.tool_calls %} 
                    {%- if (loop.first and content) or (not loop.first) %} 
                        {{- '\n' }} 
                    {%- endif %} 
                    {%- if tool_call.function %} 
                        {%- set tool_call = tool_call.function %} 
                    {%- endif %} 
                    {{- '<tool_call>\n{"name": "' }} 
                    {{- tool_call.name }} {{- '", "arguments": ' }} 
                    {%- if tool_call.arguments is string %} 
                        {{- tool_call.arguments }} 
                    {%- else %} 
                        {{- tool_call.arguments | tojson }} 
                    {%- endif %} 
                    {{- '}\n</tool_call>' }} 
                {%- endfor %} 
            {%- endif %} 
            {{- '<|im_end|>\n' }} 
            
        {%- elif message.role == "tool" %} 
            {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %} 
                {{- '<|im_start|>user' }} 
            {%- endif %} 
            {{- '\n<tool_response>\n' }} 
            {{- message.content }} 
            {{- '\n</tool_response>' }} 
            {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %} 
                {{- '<|im_end|>\n' }} 
            {%- endif %} 
        {%- endif %} 
    {%- endfor %} 

    {%- set add_generation_prompt = True %} 
    
    {%- if add_generation_prompt %} 
        {{- '<|im_start|>assistant\n' }} 
        {%- if enable_thinking is defined and enable_thinking is false %} 
            {{- '<think>\n\n</think>\n\n' }} 
        {%- endif %} 
    {%- endif %}
    """,

}
