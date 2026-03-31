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
        "top_k": 5,
        "extraction_token_threshold": 512,
    },

    # 시스템 프롬프트 및 변수 설정

    "VARIABLES": {
        "language": "Korean",
    },

    "SYSTEM_PROMPT": """
    You are a helpful assistant. The response language is {language}.
    """,

    "BIO_EXTRACTION_PROMPT": """
    # Role: Selective Memory Architect for Pet Bot

    You are a highly selective memory management system. Your goal is to distinguish between transient "small talk" and "meaningful information" that defines a user's identity and deepens the emotional bond. 

    # Scoring Objectives (Total: 15pts)
    Evaluate each potential memory in the <think> section using these criteria:

    1. **Relational Depth (Max 6pts)**
    - 1-2pts: Peripheral facts (Weather, routine chatter).
    - 3-4pts: Personal preferences & Daily routines (Food, hobbies, habits).
    - 5-6pts: Identity & Social Anchors (Life direction, affiliations like school/work, core values, and future goals).

    2. **Emotional & Appraisal Impact (Max 4pts)**
    - 1-2pts: Routine/Minor relevance (Daily events, temporary moods, short-term plans).
    - 3-4pts: High/Critical relevance (Life transitions, major milestones, changes in environment or social status).

    3. **Interactional Utility (Max 5pts)**
    - 1-2pts: Low utility; unlikely to be meaningful in the long term.
    - 3-4pts: Good conversational bridge; shows the bot is paying attention.
    - 5pts: Relational Keynote (Unique names of institutions, people, or critical triggers).

    # Storage & Filtering Rules (CRITICAL)
    - **Importance Score:** The final `importance` is the `Total Score` (Range: 1 to 15).
    - **Core Bio Rule:** Information regarding Name, Age, Job, Nationality, Sex, and MBTI is automatically assigned a **Score of 15** and marked `is_core: true`.
    - **Atomic Memory:** Extract EACH distinct fact as a separate `<bio>` block. Do not combine multiple facts.
    - **SPO Format:** Write the content as a concise `User [Predicate] [Object]` sentence. (e.g., "User likes hiking").
        
    # Output Format
    - Strictly output the following XML-style structure.
    - Even for low-score information, generate the <bio> tag (Filtering will be handled by the parser).

    <think>
    [Step-by-step scoring for each fact: Depth(?) + Emotion(?) + Utility(?) = Total Score.]
    [Check for Core Bio Rule]
    </think>

    <bio>
    content: [SPO Sentence]
    importance: [Total Score (1-15)]
    is_core: [true/false]
    </bio>

    [Repeat <bio> blocks for every identified fact in the conversation.]

    Below are the conversations between the user and you:
    """,

    "BIO_EXPLANATION_PROMPT": """
### Background Context
Refer to this shared context to keep your response personal and naturally informed:
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
