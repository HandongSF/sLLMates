from src.config import MODELS_DIR



CONFIG = {

    # 채팅 모델 설정

    "CHAT_MODEL_CONFIG": {
        # class: Llama
        "model_path": str(MODELS_DIR / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        # Model Params
        "n_gpu_layers": -1,
        "main_gpu": 1,
        "tensor_split": [0.05, 0.95],
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
        "temperature": 1.0,
        "top_p": 1,
        "min_p": 0,
        "stop": ["<|end_of_text|>", "<|eot_id|>"],
        "top_k": 20,
    },

    # 임베딩 모델 설정

    "EMBEDDING_MODEL_CONFIG": {
        "model_name": str(MODELS_DIR / "Qwen3-Embedding-0.6B"),
        "model_kwargs": {'device': 'cuda'},
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

    # RAG 설정

    "RAG_CONFIG": {
        "chunk_size": 200,
        "chunk_overlap": 50,
        "batch_size": 16,
        "retrieval_k": 5,
    },

    # 시스템 프롬프트 및 변수 설정

    "VARIABLES": {
        "language": "Korean",
    },

    "SYSTEM_PROMPT": """
    You are Llama3.1, a large language model trained by Meta, based on the Llama architecture. You are chatting with the user via the Chating app. Never use emojis unless explicitly asked to. When you receive a tool call response, use the output to format an answer to the orginal user question. The response language is {language}.
    """,

    "BIO_PROMPT": """
    You are Llama3.1, a large language model trained by Meta. 
    Your task is ONLY to extract long-term, meaningful facts about the user.

    If you find a new, stable fact about the user, you MUST output it using the EXACT format below:

    <bio>the fact</bio>
    <importance>N</importance>

    You must follow this format EXACTLY.
    No extra text before, between, or after these tags.

    Save only long-term or meaningful information (preferences, background, personality, health, goals).
    Ignore trivial or temporary details (e.g., current location, mood, filler expressions, greetings).

    Examples (follow the exact format):
    <bio>The user is allergic to peanuts.</bio>
    <importance>9</importance>

    <bio>The user's birthday is May 12.</bio>
    <importance>8</importance>

    If there is no new meaningful user fact, output nothing.

    Below are the user queries:
    """,

    "BIO_EXPLANATION_PROMPT": """
    \nBelow are important information about the user:\n""",

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
        "eos_token": "<|eot_id|>",
        "bos_token": "<|begin_of_text|>",
    },

    "CUSTOM_CHAT_TEMPLATE_VARIABLES": {
        ""
    },

    "CUSTOM_CHAT_TEMPLATE": """
    {%- if custom_tools is defined %} 
        {%- set tools = custom_tools %} 
    {%- endif %} 

    {%- if not tools_in_user_message is defined %} 
        {%- set tools_in_user_message = false %} 
    {%- endif %} 

    {%- if not date_string is defined %} 
        {%- set date_string = strftime_now('%d %b %Y') | default("26 Jul 2024", true) %} 
    {%- endif %} 

    {%- if not tools is defined %} 
        {%- set tools = none %} 
    {%- endif %} 

    {#- This block extracts the system message, so we can slot it into the right place. #} 
    {%- if messages[0]['role'] == 'system' %} 
        {%- set system_message = messages[0]['content']|trim %} 
        {%- set messages = messages[1:] %} 
    {%- else %} 
        {%- set system_message = "" %} 
    {%- endif %} 

    {#- System message + builtin tools #} 
    {{- "<|start_header_id|>system<|end_header_id|>\n\n" }} 
    {%- if builtin_tools is defined or tools is not none %} 
        {{- "Environment: ipython\n" }} 
    {%- endif %} 
    {%- if builtin_tools is defined %} 
        {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}} 
    {%- endif %} 

    {{- "Cutting Knowledge Date: December 2023\n" }} 
    {{- "Today Date: " + date_string + "\n\n" }} 
    {%- if tools is not none and not tools_in_user_message %} 
        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }} 
        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }} 
        {{- "Do not use variables.\n\n" }} 
        {%- for t in tools %} 
            {{- t | tojson(indent=4) }} 
            {{- "\n\n" }} 
        {%- endfor %} 
    {%- endif %} 

    {{- system_message }} 
    {{- "<|eot_id|>" }} 
    {#- Custom tools are passed in a user message with some extra guidance #} 
    {%- if tools_in_user_message and not tools is none %} 
        {#- Extract the first user message so we can plug it in here #} 
        {%- if messages | length != 0 %} 
            {%- set first_user_message = messages[0]['content']|trim %} 
            {%- set messages = messages[1:] %} 
        {%- else %} 
            {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }} 
        {%- endif %} 
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}} 
        {{- "Given the following functions, please respond with a JSON for a function call " }} 
        {{- "with its proper arguments that best answers the given prompt.\n\n" }} 
        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }} 
        {{- "Do not use variables.\n\n" }} 
        {%- for t in tools %} 
            {{- t | tojson(indent=4) }} 
            {{- "\n\n" }} 
        {%- endfor %} 
        {{- first_user_message + "<|eot_id|>"}} 
    {%- endif %} 

    {%- for message in messages %} 
        {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %} 
            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }} 
        {%- elif 'tool_calls' in message %} 
            {%- if not message.tool_calls|length == 1 %} 
                {{- raise_exception("This model only supports single tool-calls at once!") }} 
            {%- endif %} 
            {%- set tool_call = message.tool_calls[0].function %} 
            {%- if builtin_tools is defined and tool_call.name in builtin_tools %} 
                {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}} 
                {{- "<|python_tag|>" + tool_call.name + ".call(" }} 
                {%- for arg_name, arg_val in tool_call.arguments | items %} 
                    {{- arg_name + '="' + arg_val + '"' }} 
                    {%- if not loop.last %} 
                        {{- ", " }} 
                    {%- endif %} 
                {%- endfor %} 
                {{- ")" }} 
            {%- else %} 
                {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}} 
                {{- '{"name": "' + tool_call.name + '", ' }} 
                {{- '"parameters": ' }} 
                {{- tool_call.arguments | tojson }} 
                {{- "}" }} 
            {%- endif %} 
            {%- if builtin_tools is defined %} 
                {#- This means we're in ipython mode #} 
                {{- "<|eom_id|>" }} 
            {%- else %} 
                {{- "<|eot_id|>" }} 
            {%- endif %} 
        {%- elif message.role == "tool" or message.role == "ipython" %} 
            {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }} 
            {%- if message.content is mapping or message.content is iterable %} 
                {{- message.content | tojson }} 
            {%- else %} 
                {{- message.content }} 
            {%- endif %} 
            {{- "<|eot_id|>" }} 
        {%- endif %} 
    {%- endfor %} 

    {%- if add_generation_prompt %} 
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }} 
    {%- endif %}
    """,

}