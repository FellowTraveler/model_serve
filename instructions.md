===> Please explain Harmony, since GPT-OSS is an OpenAI model so it seems strange that they would produce a model that doesn't support OpenAI tool calls.


Harmony is a separate, lower‑level conversation format that GPT‑OSS was explicitly trained on; the “weirdness” you’re seeing is that GPT‑OSS natively speaks Harmony, not the vanilla Chat Completions JSON that llama‑server expects for tools.[1][2]

## What Harmony is

- Harmony is a **token-level message format**: conversations are rendered as `<|start|>{header}<|message|>{content}<|end|>` blocks instead of JSON messages. [1][3]  
- Each block has a role (system, developer, user, assistant, tool) plus a **channel** that indicates what kind of content it is (e.g. `analysis` for reasoning, `final` for the user-facing answer, `commentary` for tool calls).[4][1]
- GPT‑OSS models are post‑trained specifically on this format; OpenAI ships a “Harmony encoder/decoder” that takes normal messages/tools in your app and turns them into these tokens, then parses them back out.[2][3][1]

Example shape (simplified):

- Input:  
  `user: "What is 2+2?"` → rendered as `"<|start|>user<|message|>What is 2 + 2?<|end|>\n<|start|>assistant"` [1]  
- Output stream from the model: multiple assistant messages on different channels, e.g. `analysis` for chain-of-thought, then `final` with the answer.[5][1]

## How tool calling works in Harmony

Within Harmony, tools are **not** the OpenAI `tool_calls` JSON you’re used to; they’re encoded as assistant messages on specific channels with a `to=` recipient in the header.[6][1]

- You define tools in a special “Tools” section in the **developer** message using a TypeScript-like schema under a `functions` namespace.[1]
- When GPT‑OSS wants to call a tool, it emits a Harmony message such as:  
  - role: assistant  
  - channel: `commentary` (tool invocation)  
  - header includes `to=functions.get_current_weather`  
  - content is the JSON arguments.[6][1]
- After your code runs the tool, you respond with a **tool message** formatted like:  
  `<|start|>{toolname} to=assistant<|channel|>commentary<|message|>{output}<|end|>` so the model can read the result. [1]  

So GPT‑OSS absolutely supports tool calling—but only if you speak Harmony correctly and have a Harmony renderer on the client side.[7][1]

## Why this doesn’t plug straight into llama-server

The OpenAI HTTP APIs hide Harmony from you:

- You send standard `messages` + `tools` to `/v1/chat/completions` or Responses.[8][2]
- OpenAI’s backend uses the Harmony encoder to convert that into the special token format, runs GPT‑OSS, and then decodes the Harmony messages back into JSON with `tool_calls`, `choices[].message`, etc.[3][1]

llama-server and llama-swap don’t implement this encoder/decoder:

- llama-server expects either classic chat templates (ChatML‑ish) or its own tool-enabled templates from `function-calling.md` (Llama, Qwen, Mistral, etc.).[9]
- GPT‑OSS GGUFs you’re running locally still emit raw Harmony tokens, including the `to=...`, `analysis`, `final`, `commentary` channels, but llama-server treats all of that as plain text because it has no Harmony parser.[10][1]
- Consequently, llama-swap never sees proper `tool_calls` objects; it just sees strings like `<|start|>assistant<|channel|>commentary to=functions.get_foo …`. [9][10]  

That’s why it feels like “OpenAI shipped a model that doesn’t support OpenAI tools”—in reality, they shipped:

- A model family trained on Harmony.[2]
- A reference Harmony SDK/encoder for you to integrate.[3][1]
- But third‑party runtimes (llama.cpp, llama-swap) haven’t implemented that layer yet, so you must either ignore tools or hand‑roll a Harmony adapter.[11][10]

## How to think about it in your stack

Given your setup:

- GPT‑OSS **does** support tools, but only via the Harmony flow: your client must render messages+tools into Harmony and parse Harmony tool messages back.[7][1]
- llama-server’s “native tools” path is completely separate and expects its own templates; it does not understand Harmony.[12][9]
- So for now, practical options are:
  - Use GPT‑OSS through an OpenAI‑compatible service that already implements Harmony, or  
  - Treat GPT‑OSS as a reasoning/chat model in llama-swap and use Qwen/Llama/etc. for tool calls, or  
  - Implement a small Harmony encoder/decoder in front of llama-swap and drive tools yourself.

If you tell me what language your agent layer is in (Go/Rust/TS), I can sketch what a minimal “parse Harmony tool call, execute, feed back tool message” loop would look like.

Sources
[1] OpenAI Harmony Response Format https://developers.openai.com/cookbook/articles/openai-harmony/
[2] Introducing gpt-oss - OpenAI https://openai.com/index/introducing-gpt-oss/
[3] openai/harmony - GitHub https://github.com/openai/harmony
[4] GPT-oss provides a rare peek into LLM research at OpenAI. - LinkedIn https://www.linkedin.com/posts/cameron-r-wolfe-ph-d-04744a238_gpt-oss-provides-a-rare-peek-into-llm-research-activity-7363202076690022401-tduT
[5] ChatML vs Harmony: Understanding the new Format from OpenAI https://huggingface.co/blog/kuotient/chatml-vs-harmony
[6] Have you noticed weird tool-calling behaviour from GPT-OSS when ... https://www.linkedin.com/posts/tanmayb123_have-you-noticed-weird-tool-calling-behaviour-activity-7377914599469277184-Ms-9
[7] Verifying gpt-oss implementations - OpenAI Cookbook https://cookbook.openai.com/articles/gpt-oss/verifying-implementations
[8] Function calling | OpenAI API https://platform.openai.com/docs/guides/function-calling
[9] llama.cpp/docs/function-calling.md at master · ggml-org ... - GitHub https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
[10] How is everyone dealing with the OpenAI Harmony format on gpt-oss? https://www.reddit.com/r/LocalLLaMA/comments/1mj6y6j/how_is_everyone_dealing_with_the_openai_harmony/
[11] Build a Weather Assistant with OpenAI GPT-OSS and Harmony SDK ... https://vast.ai/article/build-a-weather-assistant-with-openai-gpt-oss-and-harmony-sdk-on-vast-ai
[12] bizarre Jinja bug when trying to fix Qwen3 tool calling · Issue #13516 https://github.com/ggml-org/llama.cpp/issues/13516
[13] GPT‑OSS Harmony Prompt Format Explained - YouTube https://www.youtube.com/watch?v=g8P7oiZkR_w
[14] a response format designed for GPT-OSS model series : r/LocalLLaMA https://www.reddit.com/r/LocalLLaMA/comments/1midvvn/introduce_openai_harmony_a_response_format/
[15] OpenAI introduces Harmony format for GPT models - LinkedIn https://www.linkedin.com/posts/winklerjenna_ai-agenticai-openai-activity-7361063295761457152-tigo
[16] How to Create a Dermatology Q&A Dataset with OpenAI Harmony ... https://www.firecrawl.dev/blog/creating_dermatology_dataset_with_openai_harmony_firecrawl_search
[17] - YouTube https://www.youtube.com/post/UgkxM2XbDNWxC6ksk2BDMK942QdUxC8ImGjM

===> Well I'm using a project called model_serve, composed of some python and shell scripts, that make it easy to start/stop llama-swap along with a custom config yaml file. Here's the function that starts it up:

start_llama_swap() {
    echo "Starting llama-swap on port ${LLAMA_SWAP_PORT}..."
    echo "Using: $LLAMA_SWAP_BIN"
    "$LLAMA_SWAP_BIN" --config config.yaml --listen "0.0.0.0:${LLAMA_SWAP_PORT}" &
    LLAMA_SWAP_PID=$!
    echo "llama-swap PID: $LLAMA_SWAP_PID"
}

===> I'm surprised it doesn't already support Harmony, especially if reference implementations are available. If we had one, would it be easy (does llama-swap support) adding it into an intermediary layer during input/output processing?


Yes, it’s very feasible to add a Harmony layer around llama-swap, but llama-swap itself doesn’t currently have a built‑in request/response transform hook, so the Harmony logic has to live in a small proxy in front of it (or in your client), not inside llama-swap.[1][2][3][4]

## What llama-swap gives you (and doesn’t)

- llama-swap is an OpenAI-compatible reverse proxy that: looks at `model`, ensures the right upstream server (llama-server/vLLM/etc.) is running, and forwards the OpenAI-style HTTP request/response unchanged.[2][1]
- It does not currently expose a “middleware” API for arbitrary per-request transforms; config.yaml lets you define how to start the upstream command, not how to rewrite JSON bodies.[1][2]
- So there’s no native knob like “encode to Harmony on ingress, decode from Harmony on egress” inside llama-swap today.  

## Where the Harmony encoder fits

The Harmony SDK gives you exactly what you need: functions to turn normal message objects into Harmony tokens, and to parse Harmony tokens back into structured channels (final answer, reasoning, tool calls, etc.).[3][4][5]

The natural architecture with your `model_serve` setup is:

1. Your agent / Goose → talks to a **tiny Python proxy** on some port (say 9000) using normal OpenAI `/v1/chat/completions` with tools.  
2. That proxy:  
   - Uses `openai_harmony` to encode the conversation + tool definitions into Harmony text/tokens.[4][3]
   - Wraps that as a plain chat/completions request to llama-swap, targeting the GPT‑OSS model.  
3. llama-swap → forwards to `llama-server` running GPT‑OSS, which just generates Harmony output as text.[6][7]
4. The proxy reads the streaming text, feeds it into `StreamableParser` from the Harmony SDK to recover channels, tool calls etc., and then returns a standard OpenAI-style JSON response (with `tool_calls`) back to your agent.[5][8][4]

From llama-swap’s point of view, it’s just another OpenAI client; all the Harmony stuff is hidden in your small Python process.

## Why this is probably the path of least resistance

- You don’t need to modify `model_serve` or llama-swap; your `start_llama_swap` shell function can stay exactly as it is.[2][1]
- You get to use the official Harmony encoder/decoder reference implementation (Python crate) that’s already tailored for GPT‑OSS (`HarmonyEncodingName.HARMONY_GPT_OSS`).[3][4][5]
- If llama-swap ever grows a formal “plugin” or transform layer, you can always collapse this proxy into it later; but today, putting the logic in a dedicated Python front-end is simpler than patching Go.  

## Concretely, what you’d implement

In that Python proxy:

- **Ingress (request → Harmony):**  
  - Accept `/v1/chat/completions` with `messages`, optional `tools`.  
  - Build a `Conversation` from those messages and `ToolDescription` objects from `tools`.[4][5]
  - Call `encoding.render_conversation_for_completion(...)` to get the Harmony prompt, send that as a single `user` or `system` prompt to llama-swap’s GPT‑OSS model.[3][4]
- **Egress (Harmony → response):**  
  - Stream tokens from llama-swap, pipe into `StreamableParser(encoding, role=Role.ASSISTANT)`.[8][4]
  - When the parser yields a tool-call channel, emit an OpenAI-style `tool_calls` event to your client; when it yields a `final` channel, emit the final assistant message.[8][4]

If you’d like, next step I can sketch a minimal Python FastAPI (or bare Flask) example that slots into `model_serve`: a `/v1/chat/completions` endpoint that does exactly this Harmony↔OpenAI translation and forwards to your existing llama-swap port.

Sources
[1] mostlygeek/llama-swap: Reliable model swapping for any ... - GitHub https://github.com/mostlygeek/llama-swap
[2] llama-swap command - github.com/mostlygeek/llama-swap https://pkg.go.dev/github.com/mostlygeek/llama-swap
[3] openai/harmony - GitHub https://github.com/openai/harmony
[4] OpenAI Harmony Response Format https://developers.openai.com/cookbook/articles/openai-harmony/
[5] How to run gpt-oss with Transformers - OpenAI for developers https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers/
[6] Switching from Ollama to llama-swap + llama.cpp on NixOS https://www.nijho.lt/post/llama-nixos/
[7] How to Run Multiple LLMs Locally Using Llama-Swap on a Single ... https://www.kdnuggets.com/how-to-run-multiple-llms-locally-using-llama-swap-on-a-single-server
[8] Build a Weather Assistant with OpenAI GPT-OSS and Harmony SDK ... https://vast.ai/article/build-a-weather-assistant-with-openai-gpt-oss-and-harmony-sdk-on-vast-ai
[9] compatible with llama-server docker · Issue #35 - GitHub https://github.com/mostlygeek/llama-swap/issues/35
[10] What's the most complicated project you've built with AI? - Reddit https://www.reddit.com/r/LocalLLaMA/comments/1qtj87p/whats_the_most_complicated_project_youve_built/
[11] Engineer's Guide to Local LLMs with LLaMA.cpp on Linux https://dev.to/avatsaev/pro-developers-guide-to-local-llms-with-llamacpp-qwen-coder-qwencode-on-linux-15h
[12] Refusal mechanisms: initial experiments with Llama-2-7b-chat https://www.alignmentforum.org/posts/pYcEhoAoPfHhgJ8YC
[13] Open WebUI and llama-swap works great together! Demo of setup ... https://www.reddit.com/r/LocalLLaMA/comments/1mon08l/tutorial_open_webui_and_llamaswap_works_great/
[14] llama.cpp is all you need : r/LocalLLaMA - Reddit https://www.reddit.com/r/LocalLLaMA/comments/1j417qh/llamacpp_is_all_you_need/
[15] llama.cpp releases new official WebUI : r/LocalLLaMA - Reddit https://www.reddit.com/r/LocalLLaMA/comments/1ooa342/llamacpp_releases_new_official_webui/



===> Not every model supports Harmony, so how would we know when to use the proxy? Would that be in the custom config yaml? Or could we just use it for all models indiscriminately?


 You should not run the Harmony proxy for all models; treat it as **per‑model glue for GPT‑OSS (and any other Harmony‑native families)**, and keep everything else going straight to llama-swap unchanged.[1][2][3]

## Which models need Harmony

- Harmony is explicitly designed for **GPT‑OSS models**; those were trained to emit and consume Harmony tokens (`HARMONY_GPT_OSS` encoding).[2][4][1]
- Other local models you’re using via llama-swap (Qwen, Llama 3.x, Gemma, Mistral, etc.) are trained on their own chat templates (ChatML-ish, Yi, Qwen, Llama, etc.), not on Harmony.[3][5]
- Running those non‑Harmony models through a Harmony encoder/decoder would just garble prompts and outputs, because the special tokens and structural expectations don’t match.[6][3]

So: **only GPT‑OSS (and any future explicitly “Harmony‑format” models, like GPT‑5 if exposed similarly) should be passed through the Harmony proxy.**[7][8]

## Where to decide “Harmony vs normal”

Because llama-swap itself has no notion of Harmony, the routing decision lives outside it:

- In llama-swap’s `config.yaml`, each model already has a **model ID** you pass as `model` in the OpenAI request.[9][10]
- Your proxy (or your agent layer) can inspect `model` and choose:
  - If `model` is one of your Harmony models (e.g. `gpt-oss-20b`, `gpt-oss-120b`), go through the Harmony encoder/decoder and then forward to llama-swap.[1][2]
  - Otherwise, just pass the request straight through to llama-swap with no transformation.  

You don’t need to encode this in llama-swap’s YAML; it’s easier and more flexible to keep a small mapping in your proxy.

### Simple pattern

In Python pseudo‑logic:

```python
HARMONY_MODELS = {"gpt-oss-20b", "gpt-oss-120b"}  # your llama-swap model IDs

if body["model"] in HARMONY_MODELS:
    # Harmony path: encode messages/tools to Harmony, send to llama-swap GPT-OSS
    ...
else:
    # Normal path: forward body unchanged to llama-swap on port 8000
    ...
```

You can drive this list from a small config file next to `config.yaml` if you want it declarative, but it does **not** need to live inside llama-swap.[10][11]

## Why not “just use Harmony for everything”

- The Harmony SDK and encoding tables (`HarmonyEncodingName.HARMONY_GPT_OSS`, `o200k_harmony`) are tailored for GPT‑OSS’s tokenization and special tokens.[12][1]
- Non‑Harmony models won’t understand `<|channel|>analysis` / `<|channel|>commentary` / Harmony role headers, and llama.cpp’s built‑in chat templates for them expect totally different special tokens. [5][3]  
- You could, in theory, invent a “fake Harmony‑like” template for other models, but then you’re no longer following OpenAI’s Harmony spec and you lose interoperability.[13][3]

So the safe rule: **Harmony encoder is only for models whose docs explicitly say “trained on Harmony format” (currently GPT‑OSS family).**[4][2]

## How this fits your existing `model_serve` setup

- Keep `model_serve` and your `start_llama_swap` function exactly as they are; llama-swap still exposes a single OpenAI-style endpoint.[9][10]
- Add a small Python proxy that:
  - Listens on a new port (e.g. 9000).  
  - Implements `/v1/chat/completions`.  
  - Uses the `model` field to decide: Harmony path (GPT‑OSS) vs direct pass‑through (everything else).  
- Point Goose/your agents to the proxy instead of llama-swap directly; all other infra stays untouched.

If you’d like, next step I can sketch a concrete FastAPI app with that `HARMONY_MODELS` routing and the minimum Harmony encode/decode loop for GPT‑OSS.

Sources
[1] OpenAI Harmony Response Format https://developers.openai.com/cookbook/articles/openai-harmony/
[2] openai/harmony - GitHub https://github.com/openai/harmony
[3] ChatML vs Harmony: Understanding the new Format from OpenAI https://huggingface.co/blog/kuotient/chatml-vs-harmony
[4] Introducing GPT-OSS-120B and GPT-OSS-20B with OpenAI ... https://iapp.co.th/blog/gpt-oss-120b-20b-openai-harmony
[5] llama.cpp/docs/function-calling.md at master · ggml-org ... - GitHub https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
[6] OpenAI's new open weight (Apache 2) models are really good https://simonwillison.net/2025/Aug/5/gpt-oss/
[7] GPT‑OSS Harmony Prompt Format Explained - YouTube https://www.youtube.com/watch?v=g8P7oiZkR_w
[8] GPT-oss provides a rare peek into LLM research at OpenAI. - LinkedIn https://www.linkedin.com/posts/cameron-r-wolfe-ph-d-04744a238_gpt-oss-provides-a-rare-peek-into-llm-research-activity-7363202076690022401-tduT
[9] mostlygeek/llama-swap: Reliable model swapping for any ... - GitHub https://github.com/mostlygeek/llama-swap
[10] llama-swap command - github.com/mostlygeek/llama-swap https://pkg.go.dev/github.com/mostlygeek/llama-swap
[11] Can llama-swap work without specifying the "model" field in API ... https://www.reddit.com/r/LocalLLaMA/comments/1mdufwb/can_llamaswap_work_without_specifying_the_model/
[12] openai/gpt-oss-120b · openai_harmony.HarmonyError - Hugging Face https://huggingface.co/openai/gpt-oss-120b/discussions/39
[13] Hacking GPT-OSS Harmony template with custom tokens - Reddit https://www.reddit.com/r/LocalLLaMA/comments/1nut65d/hacking_gptoss_harmony_template_with_custom_tokens/
[14] Harmony Response Format sometimes outputted when using gpt ... https://forum.langchain.com/t/harmony-response-format-sometimes-outputted-when-using-gpt-oss-120b-as-an-agent/2554
[15] Tool Use Differences Between gpt-oss-20b and o3-mini in Multi ... https://community.openai.com/t/tool-use-differences-between-gpt-oss-20b-and-o3-mini-in-multi-agent-setup/1338110
[16] Build a Weather Assistant with OpenAI GPT-OSS and Harmony SDK ... https://vast.ai/article/build-a-weather-assistant-with-openai-gpt-oss-and-harmony-sdk-on-vast-ai


===> Yes, please write up all the requirements / reasons why, as well as a concrete description of what needs to be built, so I can hand it off to the coding agent. Include any necessary relevant URLs. Make sure you explain thoroughly all your reasoning so there aren't any gaps in our logic for the decisions.


You need a small Harmony-aware proxy in front of llama-swap, used only for GPT‑OSS (and any other Harmony‑native models), while all other models keep using vanilla OpenAI-style JSON directly with llama-swap.[1][2][3]

## 1. Why this is needed

**a. GPT‑OSS is Harmony-native, llama-swap is not**

- GPT‑OSS models were trained on the **Harmony response format**; they expect prompts encoded as Harmony tokens and emit Harmony messages (roles, channels, `to=tool` headers, etc.).[3][4][1]
- The Harmony SDK exists specifically so inference providers can render prompts into Harmony and parse Harmony output back into structured messages and tool calls.[1][3]
- llama-swap is an OpenAI-compatible proxy that forwards JSON `messages`/`tools` to upstream servers; it has **no built-in understanding of Harmony** and does not transform requests.[2][5]

If you send GPT‑OSS plain ChatML-ish prompts via llama-server, you’re effectively feeding it off-distribution input; at best you get degraded behavior, and tool calling will not be structured correctly.[4][3]

**b. Harmony handles tools, reasoning, and final output**

- Harmony splits model output into separate **channels** (e.g., reasoning vs tool calls vs final answer), and tools are emitted as structured assistant messages with `to=functions.my_tool` and JSON arguments, all inside Harmony tokens.[6][4]
- The OpenAI HTTP APIs hide that: they internally render to Harmony, run the model, then parse Harmony back into normal `choices[].message`, `tool_calls`, etc.[3][4]
- When you run GPT‑OSS via llama.cpp/llama-server, that translation layer is missing; you just see raw Harmony-ish text like `<|start|>assistant<|channel|>commentary to=functions.get_weather …`. [4]  

Your agent (Goose) expects standard OpenAI tool-calling JSON, so you need something that plays the role of OpenAI’s internal Harmony layer.

**c. Why not do this inside llama-swap?**

- llama-swap’s design is “transparent OpenAI proxy + model router”: it inspects `model`, ensures the right upstream command (`llama-server ...`) is running, and forwards requests/responses verbatim.[5][2]
- It doesn’t expose a general “middleware” interface to rewrite request bodies or stream responses; `config.yaml` configures how to start servers, not how to transform JSON.[2][5]
- Extending llama-swap itself with Harmony logic is possible but significantly more invasive than adding a thin Python service in front of it.

So: the clean architecture is a **separate Harmony proxy**: your agents talk to the proxy; the proxy talks to llama-swap and takes care of Harmony only for Harmony-native models.

***

## 2. High-level architecture

**Existing stack**

- `model_serve` starts llama-swap with something like:  
  `llama-swap --config config.yaml --listen 0.0.0.0:${LLAMA_SWAP_PORT}`  
- llama-swap then:  
  - Offers `/v1/chat/completions` (and other OpenAI endpoints).[5]
  - Based on `body.model`, starts or reuses a `llama-server` process for the requested GGUF model.[2]

**Target stack**

Add a Harmony proxy in front:

1. **Harmony Proxy (Python)**  
   - Listens on a port, e.g. `localhost:9000`.  
   - Implements `/v1/chat/completions` (and optionally `/v1/responses` if you ever want that).  
   - Inspects `body.model` to decide how to handle the request.  

2. **Routing rules in the proxy**

   - Define a set of Harmony-native model IDs, e.g.:  
     `HARMONY_MODELS = {"gpt-oss-20b", "gpt-oss-120b"}` (names must match your llama-swap `config.yaml` model keys).[7][1]
   - If `body.model in HARMONY_MODELS`:  
     - Use the Harmony SDK (`openai-harmony`) to encode messages + tools into Harmony tokens.[4][1]
     - Send a **single** chat/completions request to llama-swap, targeting the GPT‑OSS model.  
     - Stream the Harmony output from llama-swap, parse it with Harmony’s streaming parser into structured assistant messages and tool calls.[1][3][4]
     - Emit a normal OpenAI-style JSON response to the caller (with `choices[].message`, `tool_calls`, etc.).  
   - Otherwise (`model` not in `HARMONY_MODELS`):  
     - Forward the request unchanged to llama-swap and stream the response back unchanged.  

3. **Clients (Goose, other agents)**  
   - Point to the Harmony proxy as if it were an OpenAI server.  
   - Continue to use standard OpenAI JSON with `messages`, `tools`, and expect normal tool-calling semantics.  

llama-swap and `model_serve` stay exactly as they are; they just see the proxy as another OpenAI client.

***

## 3. Detailed design: Harmony path (GPT‑OSS)

This is what the coding agent needs to implement.

### 3.1 Dependencies and docs

- Harmony SDK repo: https://github.com/openai/harmony[1]
- Harmony format article: https://developers.openai.com/cookbook/articles/openai-harmony[4]
- GPT‑OSS implementation notes: https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations/[3]
- llama-swap repo: https://github.com/mostlygeek/llama-swap[2]

Python package:

```bash
pip install openai-harmony
```



### 3.2 Request handling for Harmony models

**Input:** standard OpenAI `/v1/chat/completions` request:

- `model: string`  
- `messages: [{role, content, name?, tool_call_id?}]`  
- Optional `tools: [...]` + `tool_choice`.  
- Flags like `stream`, `max_tokens`, `temperature`, etc.

**Step A – Decide path**

```python
HARMONY_MODELS = {"gpt-oss-20b", "gpt-oss-120b"}  # configurable

if body["model"] in HARMONY_MODELS:
    handle_with_harmony(body)
else:
    proxy_direct_to_llama_swap(body)
```



**Step B – Build Harmony Conversation & tools**

Using `openai_harmony`:

- Load the GPT‑OSS encoding:  

  ```python
  from openai_harmony import (
      load_harmony_encoding,
      HarmonyEncodingName,
      Conversation,
      Message,
      Role,
      DeveloperContent,
      SystemContent,
      ToolDescription,
  )

  enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
  ```

[3][1]

- Convert OpenAI messages → Harmony messages:

  - Map roles:
    - `system` → `Role.SYSTEM` with `SystemContent`.[4]
    - `user` → `Role.USER`.  
    - `assistant` (without tool calls) → `Role.ASSISTANT`.  
    - Tool results (OpenAI `role="tool"`) → Harmony tool messages as per docs (these might be handled in follow-up turns).[4]
  - Include any developer instructions as `Role.DEVELOPER` with `DeveloperContent`.[4]

- Convert OpenAI `tools` → Harmony `ToolDescription` definitions:

  - For each tool with JSON schema: create a `ToolDescription` / “functions” entry as per Harmony examples.[3][4]
  - Attach tools to the `Conversation` (usually in a developer message).  

At the end, you have a `Conversation` object that fully represents the request in Harmony’s higher-level data structures.

**Step C – Render Harmony prompt for GPT‑OSS**

Use the SDK to get the token sequence / string you need to send to llama-swap:

```python
prompt_tokens = enc.render_conversation_for_completion(
    convo,
    Role.ASSISTANT,
)
```



Depending on how you configure llama-server, you either:

- Send `prompt_tokens` as a single `user` message (`messages=[{"role": "user", "content": prompt_tokens}]`) to llama-swap’s `/v1/chat/completions` targeting your GPT‑OSS model, or  
- Configure `llama-server` for GPT‑OSS in “raw” mode and treat `prompt_tokens` as the whole prompt text.

Key: llama-swap is just forwarding; the GPT‑OSS model sees Harmony-format tokens exactly as intended.[3][4]

**Step D – Invoke llama-swap, streaming output**

- POST to `http://llama-swap-host:PORT/v1/chat/completions` with `stream: true` for GPT‑OSS.[5]
- Read the text tokens from the stream (they will be Harmony messages, including special tokens like `<|start|>`, `<|channel|>analysis`, etc.). [4][1]  

Feed that text into Harmony’s streaming parser.

**Step E – Parse Harmony output into assistant messages / tools**

Use `StreamableParser` from `openai_harmony`:

- Initialize:  

  ```python
  from openai_harmony import StreamableParser

  parser = StreamableParser(enc, role=Role.ASSISTANT)
  ```

[1][4]

- For each chunk of text from llama-swap, pass it into the parser.  
- The parser yields structured message events:  
  - Assistant messages on channel `analysis` (chain-of-thought).[6][4]
  - Assistant messages on channel `final` (user-facing answer).  
  - Assistant messages on a tool channel with `to=functions.<name>` and JSON args (tool calls).  

You must decide which channels to expose:

- For privacy, you probably **drop `analysis`** and only surface `final` and tool calls in the OpenAI response.[3][4]

**Step F – Construct the OpenAI-style response**

- If the parsing yields tool calls:  
  - Turn them into `tool_calls` entries under `choices[0].message.tool_calls`, as in normal OpenAI function calling.[8][4]
- When you receive a `final` channel message:  
  - Use that as `choices[0].message.content`.  

Return to the client:

```json
{
  "id": "...",
  "object": "chat.completion",
  "model": "gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls" | "stop",
      "message": {
        "role": "assistant",
        "content": "...",
        "tool_calls": [ ... ]   // if any
      }
    }
  ]
}
```

Exactly what Goose expects.

Streaming mode: your proxy should emit SSE `` chunks that mirror OpenAI’s streamed deltas (`choices[].delta`), assembling them from Harmony parser events.

***

## 4. Detailed design: normal path (non-Harmony models)

For any `model` **not** in `HARMONY_MODELS`:

- Simply forward the request body as-is to llama-swap’s `/v1/chat/completions`.[5][2]
- Stream the response bytes back unmodified.  

This preserves existing behavior for Qwen, Llama 3.x, Gemma, Mistral, etc., which already rely on llama-server’s own chat templates and tool-calling implementations.[9][10]

***

## 5. Configuration and integration points

**a. Harmony model list**

- Keep a small, explicit list of models that should use Harmony, e.g. in a YAML/JSON config next to `config.yaml`:  

  ```yaml
  harmony_models:
    - gpt-oss-20b
    - gpt-oss-120b
  ```

  Or hard-code for now.[11][7]

- Ensure these IDs match llama-swap `models:` keys so that when the proxy sends `model: "gpt-oss-20b"` to llama-swap, it picks the right upstream config.[2][5]

**b. llama-swap config**

- No changes needed for Harmony logic.  
- For GPT‑OSS models, configure `cmd` to run `llama-server` with appropriate args (no Harmony-specific flags; Harmony happens in the proxy).[2][3]

**c. model_serve integration**

- Keep your `start_llama_swap` function unchanged.  
- Add a similar function to start the Harmony proxy (e.g. a Python FastAPI/Starlette app).  
- Point Goose/OpenAI clients at the Harmony proxy port instead of llama-swap’s port.

***

## 6. Edge cases and reasoning about them

**a. Why not wrap all models in Harmony?**

- Harmony encodings (`HARMONY_GPT_OSS`) and special tokens are designed for GPT‑OSS; other models weren’t trained on them.[12][11]
- If you wrapped Qwen, Llama, etc., you’d be giving them prompts full of unknown tokens (`<|channel|>analysis`, etc.), which harms quality and likely breaks tool behavior. [12][9]  
- Harmony is explicitly advertised as “for GPT‑OSS model series” in docs and third-party writeups.[13][7][12]

Thus the explicit `HARMONY_MODELS` list is necessary and correct.

**b. Why not rely on llama.cpp’s function-calling instead?**

- llama.cpp’s function-calling templates are designed for specific families (Llama 3.x, Qwen 2.5/3, Mistral, Functionary, etc.). GPT‑OSS is not in that list today.[9][3]
- GPT‑OSS’s best behavior (especially for tools) assumes Harmony prompts; trying to retrofit it into llama.cpp’s generic function-calling grammar gives worse performance and inconsistent tool outputs.[14][3]

So the Harmony layer is the “correct” way to preserve GPT‑OSS’s intended behavior.

**c. Why a separate process instead of patching llama-swap?**

- Clean separation of concerns: llama-swap continues to do model routing and OpenAI proxying; Harmony proxy focuses on a single responsibility (Harmony↔OpenAI translation).[15][5][2]
- Much easier to iterate and restart independently; no need to fork or maintain your own llama-swap.  
- Other tools (e.g., Roo-Code, Harbor) are taking a similar approach—adding Harmony at the edge, not inside the router.[16][17]

***

## 7. Deliverables for the coding agent

The coding agent should build:

1. **A small Python web service** (FastAPI or similar) that:  
   - Exposes `/v1/chat/completions`.  
   - Reads `model` from the JSON body.  
   - Has a configurable set `HARMONY_MODELS`.  
   - For models in `HARMONY_MODELS`, uses `openai_harmony` to:  
     - Build a `Conversation` + tools.[1][4]
     - Render a Harmony prompt for GPT‑OSS.  
     - Call llama-swap’s `/v1/chat/completions` (streaming).  
     - Parse Harmony output into OpenAI-style assistant messages and tool_calls.  
     - Return OpenAI-style JSON (and streaming SSE) to the client.  
   - For other models, just proxies to llama-swap unchanged.

2. **Configuration file** for the proxy:  
   - `llama_swap_base_url` (e.g., `http://127.0.0.1:8000`).  
   - `harmony_models: [gpt-oss-20b, gpt-oss-120b]`.  

3. **Integration notes** for `model_serve`:  
   - Start llama-swap as before.  
   - Start Harmony proxy (e.g. `uvicorn harmony_proxy:app --port 9000`).  
   - Point Goose/OpenAI clients to `http://localhost:9000` instead of llama-swap.

If you want, the next step can be a concrete FastAPI implementation sketch (full file) that the coding agent can refine, including streaming handling and a placeholder OpenAI→Harmony message mapping.

Sources
[1] openai/harmony - GitHub https://github.com/openai/harmony
[2] mostlygeek/llama-swap: Reliable model swapping for any ... - GitHub https://github.com/mostlygeek/llama-swap
[3] Verifying gpt-oss implementations - OpenAI for developers https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations/
[4] OpenAI Harmony Response Format https://developers.openai.com/cookbook/articles/openai-harmony/
[5] llama-swap command - github.com/mostlygeek/llama-swap https://pkg.go.dev/github.com/mostlygeek/llama-swap
[6] OpenAI introduces Harmony format for GPT models - LinkedIn https://www.linkedin.com/posts/winklerjenna_ai-agenticai-openai-activity-7361063295761457152-tigo
[7] What is GPT OSS Harmony Response Format? https://cobusgreyling.substack.com/p/what-is-gpt-oss-harmony-response
[8] Function calling | OpenAI API https://platform.openai.com/docs/guides/function-calling
[9] llama.cpp/docs/function-calling.md at master · ggml-org ... - GitHub https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
[10] Engineer's Guide to Local LLMs with LLaMA.cpp on Linux https://dev.to/avatsaev/pro-developers-guide-to-local-llms-with-llamacpp-qwen-coder-qwencode-on-linux-15h
[11] Introducing GPT-OSS-120B and GPT-OSS-20B with OpenAI ... https://iapp.co.th/blog/gpt-oss-120b-20b-openai-harmony
[12] ChatML vs Harmony: Understanding the new Format from OpenAI https://huggingface.co/blog/kuotient/chatml-vs-harmony
[13] Harmony: OpenAI's response format for its open-weight model series https://news.ycombinator.com/item?id=44799869
[14] OpenAI's new open weight (Apache 2) models are really good https://simonwillison.net/2025/Aug/5/gpt-oss/
[15] Harmony Architecture for GPT Agent Ecosystems (FDL ... - GitHub https://github.com/openai/openai-cookbook/issues/1855
[16] Add Harmony AI Provider Support for GPT-OSS Models - GitHub https://github.com/RooCodeInc/Roo-Code/issues/10935
[17] 2.3.40 Satellite llamaswap · av/harbor Wiki - GitHub https://github.com/av/harbor/wiki/2.3.40-Satellite-llamaswap
[18] a response format designed for GPT-OSS model series : r/LocalLLaMA https://www.reddit.com/r/LocalLLaMA/comments/1midvvn/introduce_openai_harmony_a_response_format/
[19] OpenAI - GitHub https://github.com/OPENAI
[20] kooshi/llama-swappo: llama-swap + a minimal ollama compatible api https://github.com/kooshi/llama-swappo/


===> Yes, please do so. And yes we definitely want a "harmony models" yaml file. And we definitely want to maintain all streaming functionality. Also, I believe llama-swap supports anthropic's messages api, so we need to support that as well (basically support ANY api that llama-swap supports, we should be able to pass thru).


You want a Harmony-aware proxy that sits in front of llama-swap, applies Harmony only for GPT‑OSS models, preserves streaming, and passes through all other APIs/endpoints (OpenAI and Anthropic) unchanged.[1][2][3]

Below is a concrete design the coding agent can implement.

***

## 1. Overall goals and constraints

- **Goal:** Use GPT‑OSS with proper Harmony prompts and tool calling while keeping llama-swap as-is.[3][4][5]
- **Constraints:**
  - llama-swap is an OpenAI/Anthropic-compatible router with no request/response middleware hooks; it just routes based on `model`.[2][1]
  - GPT‑OSS is trained **natively on Harmony format**; quality and tool use are best if you encode/decode Harmony correctly.[4][6][3]
  - Other models (Qwen, Llama, Gemma, Mistral, etc.) are *not* Harmony models and must **not** be run through Harmony encoding.[6][7]
  - Streaming must be preserved end-to-end (OpenAI-style SSE and Anthropic Messages streaming).[8][9][3]
  - llama-swap already supports OpenAI chat completions and Anthropic Messages (`/v1/messages`) and other endpoints.[9][2]

**Conclusion:** Build an external proxy (“Harmony proxy”) in front of llama-swap. The proxy decides per-request whether to apply Harmony (GPT‑OSS) or to passthrough (everything else) and supports all endpoints llama-swap supports.

***

## 2. High-level architecture

### Components

1. **llama-swap** (unchanged)  
   - Runs on e.g. `http://127.0.0.1:8000`.  
   - Configured with models including GPT‑OSS and others.[1][2]

2. **Harmony proxy** (new, Python)  
   - Runs on e.g. `http://127.0.0.1:9000`.  
   - Exposes:
     - OpenAI endpoints: `/v1/chat/completions`, `/v1/completions`, etc.  
     - Anthropic endpoint: `/v1/messages`.[2][9]
   - For each incoming request:
     - Reads `model`.  
     - If `model` is in a configurable Harmony list, uses Harmony encode/decode for GPT‑OSS.  
     - Otherwise, forwards the request and stream **byte-for-byte** to llama-swap.  

3. **Harmony model config file** (YAML)  
   - Lists model IDs that must use Harmony.  
   - Example:  

     ```yaml
     # harmony_models.yaml
     harmony_models:
       - gpt-oss-20b
       - gpt-oss-120b
     ```

     These names must match the `model` keys used in llama-swap’s `config.yaml`.[10][1]

4. **Clients (Goose, etc.)**  
   - Pointed at the Harmony proxy (`http://127.0.0.1:9000`) instead of llama-swap directly.  
   - Continue using:
     - OpenAI-style APIs (chat completions, tools).  
     - Anthropic Messages API (`/v1/messages`) if desired.[9]

***

## 3. Harmony path: GPT‑OSS via `/v1/chat/completions`

This is the core part to implement; everything else is passthrough.

### 3.1 Dependencies and references

- Harmony SDK: https://github.com/openai/harmony[4]
- Harmony docs: https://developers.openai.com/cookbook/articles/openai-harmony[3]
- GPT‑OSS verification: https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations/[5]
- llama-swap: https://github.com/mostlygeek/llama-swap[1]

Install:

```bash
pip install openai-harmony httpx fastapi uvicorn
```



### 3.2 Routing decision

In the proxy:

```python
import yaml

with open("harmony_models.yaml") as f:
    cfg = yaml.safe_load(f)
HARMONY_MODELS = set(cfg.get("harmony_models", []))

def is_harmony_model(model: str) -> bool:
    return model in HARMONY_MODELS
```

On `/v1/chat/completions`:

```python
if is_harmony_model(body["model"]):
    return handle_chat_with_harmony(body, stream=body.get("stream", False))
else:
    return proxy_chat_to_llama_swap(body)  # transparent pass-through
```

### 3.3 Building the Harmony conversation

Use `openai_harmony` types:

```python
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ToolDescription,
    StreamableParser,
)
```



Load GPT‑OSS encoding once at startup:

```python
ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
```

**Mapping OpenAI messages → Harmony:**

- OpenAI roles → Harmony roles:
  - `system` → `Role.SYSTEM` + `SystemContent`.[3]
  - `user` → `Role.USER`.  
  - `assistant` → `Role.ASSISTANT` (no tools).  
  - `tool` → will map to Harmony tool messages in subsequent calls; for the initial implementation, you can treat them as synthetic user/developer messages or skip until you implement full turn-by-turn loops.  

- If you use a “developer” role in your app, map that to `Role.DEVELOPER` + `DeveloperContent`.[3]

Example sketch:

```python
def openai_messages_to_harmony(messages):
    hm = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")

        if role == "system":
            hm.append(Message(
                role=Role.SYSTEM,
                content=SystemContent(text=content),
            ))
        elif role == "user":
            hm.append(Message(
                role=Role.USER,
                content=content,
            ))
        elif role == "assistant":
            hm.append(Message(
                role=Role.ASSISTANT,
                content=content,
            ))
        # tool messages can be mapped later as needed
    return hm
```

**Mapping tools → Harmony ToolDescription:**

OpenAI tools with JSON schemas convert to Harmony descriptions:

```python
def openai_tools_to_harmony(tools):
    htools = []
    for t in tools:
        if t["type"] != "function":
            continue
        fn = t["function"]
        htools.append(ToolDescription(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {}),
        ))
    return htools
```

You then construct the conversation:

```python
def build_conversation(body):
    messages = openai_messages_to_harmony(body["messages"])
    tools = openai_tools_to_harmony(body.get("tools", []))
    convo = Conversation(messages=messages, tools=tools)
    return convo
```

### 3.4 Rendering Harmony for GPT‑OSS

Render the conversation into a completion prompt for an assistant turn:

```python
def render_harmony_prompt(convo):
    # we want GPT-OSS to act as assistant
    return ENC.render_conversation_for_completion(convo, Role.ASSISTANT)
```



Then send this to llama-swap as a simple chat/completions call, e.g. with a single `user` message:

```python
import httpx

async def call_llama_swap_harmony(prompt_text: str, model: str, stream: bool):
    async with httpx.AsyncClient(timeout=None) as client:
        req = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "stream": stream,
        }
        resp = await client.post(
            f"{LLAMA_SWAP_BASE}/v1/chat/completions",
            json=req,
            timeout=None,
        )
        return resp
```

Note: for streaming, use `client.stream` and iterate over `aiter_text()` to feed into Harmony’s parser.

### 3.5 Parsing Harmony output and re-emitting OpenAI responses

Use `StreamableParser` to turn Harmony text into structured events.[3]

Initialize the parser:

```python
parser = StreamableParser(ENC, role=Role.ASSISTANT)
```

As you receive text chunks from llama-swap:

```python
for chunk in upstream_stream:
    parser.process(chunk)
    # check parser.current_channel, parser.last_content_delta,
    # parser.current_recipient, etc.
```



You need logic like:

- Ignore `analysis` channel (or keep it internal).[6]
- For `channel == "final"` with role `assistant` → treat `last_content_delta` as assistant visible content.  
- For tool calls:
  - `channel` used for tool calls (often `commentary`), with `current_recipient` something like `functions.get_weather`.[11][3]
  - `last_content_delta` contains JSON arguments.  

From these, construct OpenAI-style deltas for streaming:

```python
# Example for streaming "final" content:
sse_chunk = {
    "id": "chatcmpl-...",
    "object": "chat.completion.chunk",
    "model": body["model"],
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": parser.last_content_delta,
            },
            "finish_reason": None,
        }
    ]
}
```

For tool calls, map Harmony tool message → OpenAI `tool_calls` with function name and arguments.[12][3]

You can follow the pattern of `parse_harmony_response` in the Vast.ai example, which reconstructs channels and tool calls from parser state:[11]

***

## 4. Passthrough path: all other OpenAI endpoints

For any **non-Harmony** model on `/v1/chat/completions`:

- Forward the request JSON and stream directly to llama-swap, without touching the body.[2][1]
- For streaming:
  - Use `httpx.AsyncClient.stream` and relay SSE chunks exactly as received.  
  - Don’t decode/encode; treat as bytes.

For other OpenAI endpoints (`/v1/completions`, `/v1/embeddings`, `/v1/rerank`, etc. if you need them), use the same passthrough behavior:  
- The proxy only needs to be Harmony-aware for `/v1/chat/completions` when `model in HARMONY_MODELS`.  
- Everything else is generic HTTP proxying.

***

## 5. Anthropic Messages API support (`/v1/messages`)

llama.cpp and llama-swap now support Anthropic’s Messages API with streaming and tools.[8][9][2]

**Goal:** The proxy should support the same endpoints llama-swap exposes, so you can point Anthropic-compatible clients (e.g. Claude Code) at it and still use GPT‑OSS (if you ever wanted) or other models via llama-swap.

**Design:**

- Expose `/v1/messages` on the proxy.  
- For now, **do not** try to apply Harmony to Anthropic requests; treating GPT‑OSS as Anthropic-native would require a separate mapping not covered by Harmony docs.[13][3]
- Therefore:
  - For `/v1/messages`, always passthrough to llama-swap, regardless of model.  
  - Preserve Anthropic SSE event types (`message_start`, `content_block_delta`, etc.) as-is.[8][9]

Reasoning:

- Harmony is specified for OpenAI’s message model; not defined for Anthropic’s `messages` format.[6][3]
- llama-server already implements Anthropic Messages by internally mapping to its own OpenAI-like pipeline.[9][8]
- You gain no clear benefit from mixing Harmony with Anthropic here and would risk undefined behavior.

So: Anthropic endpoints = pure proxy.

***

## 6. Proxy skeleton (FastAPI-style)

This is a simplified skeleton for the coding agent to expand:

```python
# harmony_proxy.py
import os
import yaml
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ToolDescription,
    StreamableParser,
)

LLAMA_SWAP_BASE = os.environ.get("LLAMA_SWAP_BASE", "http://127.0.0.1:8000")

with open("harmony_models.yaml") as f:
    cfg = yaml.safe_load(f)
HARMONY_MODELS = set(cfg.get("harmony_models", []))

ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

app = FastAPI()

def is_harmony_model(model: str) -> bool:
    return model in HARMONY_MODELS

# ---------- OpenAI Chat Completions ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model")
    stream = bool(body.get("stream"))

    if is_harmony_model(model):
        return await handle_chat_with_harmony(body, stream=stream)
    else:
        return await proxy_openai_endpoint("/v1/chat/completions", body, stream=stream)

async def proxy_openai_endpoint(path: str, body: dict, stream: bool):
    async with httpx.AsyncClient(timeout=None) as client:
        if stream:
            upstream = client.stream("POST", f"{LLAMA_SWAP_BASE}{path}", json=body, timeout=None)
            async def iter_stream():
                async with upstream as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
            return StreamingResponse(iter_stream(), media_type="text/event-stream")
        else:
            resp = await client.post(f"{LLAMA_SWAP_BASE}{path}", json=body, timeout=None)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

# Implement openai_messages_to_harmony, openai_tools_to_harmony, build_conversation, render_harmony_prompt here.

async def handle_chat_with_harmony(body: dict, stream: bool):
    convo = build_conversation(body)
    prompt = render_harmony_prompt(convo)

    llama_req = {
        "model": body["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,  # always stream from upstream, we can aggregate if client doesn't want stream
    }

    async with httpx.AsyncClient(timeout=None) as client:
        upstream = client.stream("POST", f"{LLAMA_SWAP_BASE}/v1/chat/completions", json=llama_req, timeout=None)

        if stream:
            async def iter_sse():
                parser = StreamableParser(ENC, role=Role.ASSISTANT)
                async with upstream as r:
                    async for raw_chunk in r.aiter_text():
                        # feed Harmony text into parser
                        parser.process(raw_chunk)
                        # from parser state, yield OpenAI-style SSE chunks
                        for sse_data in harmony_state_to_openai_deltas(parser):
                            yield f" {sse_data}\n\n"
                yield " [DONE]\n\n"
            return StreamingResponse(iter_sse(), media_type="text/event-stream")
        else:
            # Non-streaming: consume all, parse final message, return JSON
            async with upstream as r:
                text = ""
                async for raw_chunk in r.aiter_text():
                    text += raw_chunk
            parser = StreamableParser(ENC, role=Role.ASSISTANT)
            parser.process(text)
            completion = harmony_state_to_openai_final(parser, body["model"])
            return JSONResponse(content=completion)

# ---------- Anthropic Messages passthrough ----------

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    stream = bool(body.get("stream"))
    return await proxy_openai_endpoint("/v1/messages", body, stream=stream)
```

Helper functions the agent must fill in:

- `openai_messages_to_harmony`  
- `openai_tools_to_harmony`  
- `build_conversation`  
- `render_harmony_prompt`  
- `harmony_state_to_openai_deltas(parser)`  
- `harmony_state_to_openai_final(parser, model)`  

`harmony_state_to_openai_*` use parser state (`current_channel`, `last_content_delta`, `current_recipient`, etc.) to map to OpenAI style output; they can draw on examples in the Harmony docs and the Vast.ai blog parser.[13][11][3]

***

## 7. Summary of requirements for the coding agent

1. **YAML config**: `harmony_models.yaml` listing Harmony-native models.[10][4]
2. **Proxy service**:
   - Python (FastAPI/Starlette), using `openai-harmony`, `httpx`.[4][3]
   - Supports:
     - `/v1/chat/completions` (OpenAI) with Harmony path for `HARMONY_MODELS`, passthrough for others.  
     - `/v1/messages` (Anthropic) as pure passthrough.[9]
   - Maintains streaming SSE for both OpenAI and Anthropic endpoints.[8][3]
3. **Harmony logic**:
   - Message + tool mapping to `Conversation` and `ToolDescription`.[4][3]
   - Rendering with `HarmonyEncodingName.HARMONY_GPT_OSS`.[3]
   - Streaming parsing with `StreamableParser` and mapping to OpenAI-style deltas and final responses.[11][13][3]
4. **Integration**:
   - `model_serve` still starts llama-swap.  
   - Add a new start function to launch the proxy with `uvicorn harmony_proxy:app --port 9000`.  
   - Point Goose/clients to the proxy port and keep using standard OpenAI/Anthropic APIs.

If you want, we can next iterate specifically on `harmony_state_to_openai_deltas` / `final`, to define exactly how to map channels and tool calls into OpenAI-compatible tool-calling events.

Sources
[1] mostlygeek/llama-swap: Reliable model swapping for any ... - GitHub https://github.com/mostlygeek/llama-swap
[2] https://github.com/mostlygeek/llama-swap | Ecosyste.ms: Awesome https://awesome.ecosyste.ms/projects/github.com%2Fmostlygeek%2Fllama-swap
[3] OpenAI Harmony Response Format https://developers.openai.com/cookbook/articles/openai-harmony/
[4] openai/harmony - GitHub https://github.com/openai/harmony
[5] Verifying gpt-oss implementations - OpenAI for developers https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations/
[6] ChatML vs Harmony: Understanding the new Format from OpenAI https://huggingface.co/blog/kuotient/chatml-vs-harmony
[7] llama.cpp/docs/function-calling.md at master · ggml-org ... - GitHub https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
[8] New in llama.cpp: Anthropic Messages API - Hugging Face https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp
[9] https://raw.githubusercontent.com/ggml-org/llama.c... https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
[10] What is GPT OSS Harmony Response Format? https://cobusgreyling.substack.com/p/what-is-gpt-oss-harmony-response
[11] Build a Weather Assistant with OpenAI GPT-OSS and Harmony SDK ... https://vast.ai/article/build-a-weather-assistant-with-openai-gpt-oss-and-harmony-sdk-on-vast-ai
[12] Function calling | OpenAI API https://platform.openai.com/docs/guides/function-calling
[13] stream_harmony - vLLM https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/stream_harmony/
[14] New in llama.cpp: Anthropic Messages API : r/LocalLLaMA - Reddit https://www.reddit.com/r/LocalLLaMA/comments/1qhaq21/new_in_llamacpp_anthropic_messages_api/
[15] kooshi/llama-swappo: llama-swap + a minimal ollama compatible api https://github.com/kooshi/llama-swappo/
[16] Findings from a pilot Anthropic–OpenAI alignment evaluation exercise https://openai.com/index/openai-anthropic-safety-evaluation/
[17] Claude Code with Anthropic API compatibility · Ollama Blog - LinkedIn https://www.linkedin.com/posts/mateusz-klatt_claude-code-with-anthropic-api-compatibility-activity-7419870554540920833-8d9B
[18] Chat Templates - by Asimov's Addendum and Sruly Rosenblat https://asimovaddendum.substack.com/p/chat-templates
[19] kultivator-consulting/goharmony - GitHub https://github.com/kultivator-consulting/goharmony
[20] Have you noticed weird tool-calling behaviour from GPT-OSS when ... https://www.linkedin.com/posts/tanmayb123_have-you-noticed-weird-tool-calling-behaviour-activity-7377914599469277184-Ms-9


===> yes, please do so.

Here’s how to map Harmony parser state → proper OpenAI tool-calling deltas and final messages, in a way your coding agent can implement directly.

All examples assume the proxy design we already discussed.

***

## 1. What we’re mapping

Harmony streaming parser exposes, on each `process(chunk)`:

- `current_role` – typically `assistant` or a tool name.[1]
- `current_channel` – e.g. `"analysis"`, `"final"`, `"commentary"`.[2][1]
- `current_recipient` – e.g. `"functions.get_current_weather"` when calling a tool.[3][1]
- `last_content_delta` – the latest text added to the current message.[1]
- `current_content` – full accumulated content for the current message.[1]

We must turn this into:

- **Streaming** OpenAI chat completion chunks (`object: "chat.completion.chunk"` with `choices[].delta`).  
- **Non-streaming** OpenAI chat completion responses (`object: "chat.completion"` with `choices[].message`).  

We also need to:

- Ignore `analysis` for user-visible output (drop chain-of-thought).[4]
- Interpret `commentary` + `recipient=functions.foo` as a tool call.[3][1]
- Use `final` channel (or equivalent) as the assistant’s visible answer.[5][2]

***

## 2. Streaming: `harmony_state_to_openai_deltas(parser)`

This helper is called after each `parser.process(chunk)` and can emit zero or more **SSE data payloads** (as JSON strings) to send to the client.

### 2.1 Conceptual rules

1. **Analysis channel**  
   - If `current_channel == "analysis"`:  
     - Do not emit to the client.  
     - This preserves the model’s reasoning internally but doesn’t leak it.[4][1]

2. **Tool call channel**  
   - If `current_channel == "commentary"` and `current_recipient` starts with `"functions."`:[3][1]
     - Treat this as an in-progress tool call:
       - Tool name = `current_recipient.split(".", 1)[1]`.  
       - Arguments = JSON built from `current_content` (or you can wait until you see a call terminator; in practice, parsing once per message boundary is safest).[5][1]
     - Emit an OpenAI-style `tool_calls` delta.

3. **Final channel**  
   - If `current_channel == "final"` and `current_role == "assistant"`:
     - Treat `last_content_delta` as user-visible assistant text and emit as `delta.content`.[2][1]

4. **Finish reason**  
   - After Harmony indicates the end of the assistant’s final message (parser moves out of `final` or stream ends), emit a chunk with `"finish_reason": "stop"` and then `"[DONE]"`.[5][1]

### 2.2 Example implementation sketch

This is deliberately simplified but shows the core logic. The coding agent can refine edge cases like partial JSON.

```python
import json
import uuid

def harmony_state_to_openai_deltas(parser):
    """
    Generator: inspect parser state after each .process(chunk)
    and yield zero or more JSON-serializable dicts representing
    OpenAI chat.completion.chunk objects.
    """
    deltas = []

    role = parser.current_role
    channel = parser.current_channel
    recipient = parser.current_recipient
    delta_text = parser.last_content_delta or ""

    # 1) Drop analysis channel
    if channel == "analysis":
        return deltas  # no user-visible output

    # 2) Tool call in commentary channel
    if channel == "commentary" and recipient and recipient.startswith("functions."):
        fn_name = recipient.split(".", 1)[1]

        # We assume Harmony is building a single JSON object for args in this message.
        # You can also buffer until message end; parser.current_content holds full text.
        args_text = parser.current_content or ""
        tool_call_id = f"call_{uuid.uuid4().hex}"

        delta = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "model": "gpt-oss",  # caller can override
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": fn_name,
                                    "arguments": args_text,
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        deltas.append(delta)
        return deltas

    # 3) Final channel => assistant visible text
    if channel == "final" and role == "assistant" and delta_text:
        delta = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "model": "gpt-oss",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": delta_text,
                    },
                    "finish_reason": None,
                }
            ],
        }
        deltas.append(delta)
        return deltas

    # 4) Any other channel/role => no visible delta by default
    return deltas
```

Usage inside your `iter_sse()`:

```python
async def iter_sse():
    parser = StreamableParser(ENC, role=Role.ASSISTANT)
    async with upstream as r:
        async for raw_chunk in r.aiter_text():
            parser.process(raw_chunk)
            for delta in harmony_state_to_openai_deltas(parser):
                yield f" {json.dumps(delta)}\n\n"
    # Emit final stop chunk
    stop_chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f" {json.dumps(stop_chunk)}\n\n"
    yield " [DONE]\n\n"
```

This preserves streaming semantics and maps Harmony channels to OpenAI deltas. vLLM’s `serving_chat_stream_harmony` module uses a similar approach (delta extraction from parser state).[6][5]

***

## 3. Non-streaming: `harmony_state_to_openai_final(parser, model)`

For non-streaming, you consume all text, call `parser.process(text)` once (or incrementally then inspect final state), and then:

- Build one OpenAI `choices[0].message`.  
- Optionally expose tool calls if the last message was a tool invocation.[7][1]

### 3.1 Conceptual rules

1. Collect all **final channel** assistant content for the answer.[4]
2. Detect if the last assistant message in `commentary` with a tool recipient constitutes a tool call; if so:
   - Fill `message.tool_calls` similarly to the streaming case.[1][3]

Given Harmony’s spec, parsing non-streaming is easier if you call `encoding.parse_messages_from_completion_tokens` or similar, but since we’re sticking to `StreamableParser` for both cases, we can inspect `parser.current_channel` and `parser.current_content` after the full text is processed.[2][1]

### 3.2 Example implementation sketch

Simplified: assume the last message we care about is either:

- `channel == "final", role == assistant` (answer), or  
- `channel == "commentary", recipient functions.foo` (tool call).  

```python
def harmony_state_to_openai_final(parser, model: str):
    """
    Build a non-streaming OpenAI chat completion from final parser state.
    For now, we use the last assistant final message as content, and
    any last tool call as tool_calls.
    """
    role = parser.current_role
    channel = parser.current_channel
    recipient = parser.current_recipient
    content = parser.current_content or ""

    message = {
        "role": "assistant",
        "content": None,
    }
    tool_calls = None
    finish_reason = "stop"

    # Tool call case
    if channel == "commentary" and recipient and recipient.startswith("functions."):
        fn_name = recipient.split(".", 1)[1]
        args_text = content

        tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": args_text,
                },
            }
        ]
        # message.content can be empty; client sees this as "tool_calls" finish
        finish_reason = "tool_calls"

    # Final visible assistant content
    elif channel == "final" and role == "assistant":
        message["content"] = content

    # Compose response
    if tool_calls:
        message["tool_calls"] = tool_calls

    completion = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    return completion
```

For more nuanced behavior (e.g., multiple tool calls, multiple final messages), your coding agent can:

- Track a list of messages across the life of the parser, not just the last one.  
- Use that list to construct a richer `message.content` (e.g., concatenating all `final` channel segments).  

The Vast.ai Harmony weather assistant example does exactly this type of accumulation across channels and messages.[5]

***

## 4. Why these mappings are logically sound

- Harmony spec: tool calls are emitted on a **commentary** channel with a `recipient` like `functions.get_current_weather` and a constrained JSON message body.[3][1]
- OpenAI tool calling spec: tools are expressed as `tool_calls` with `function.name` and `function.arguments` JSON string.[7]
- Our mapping:
  - `recipient` → `function.name` (strip `functions.` namespace).  
  - Harmony JSON body → `function.arguments` as a string.  
  - `commentary` channel tool messages → `finish_reason: "tool_calls"` in non-streaming responses or `delta.tool_calls` in streaming.  

- Harmony’s `analysis` channel holds chain-of-thought; we explicitly drop it for user-visible output, matching OpenAI guidance and vLLM’s helper utilities.[4][1]
- Harmony’s `final` channel is intended as user-facing output; we map its content verbatim to assistant `content`.[2][5]

vLLM and other projects that implemented Harmony (e.g., vLLM’s `stream_harmony` and `harmony_utils`) follow this same structure: parse channels and recipients from Harmony, drop `analysis`, promote `final` to response content, and convert `commentary` + `recipient` into tool invocations.[6][4][5]

***

## 5. What the coding agent should implement next

1. Add **parser-state tracking** around `StreamableParser`:
   - Optionally keep an internal list of events (channel, recipient, full content) for more robust non-streaming handling.[1]

2. Implement the two helpers in the proxy:
   - `harmony_state_to_openai_deltas(parser)` (streaming) – as above, but parameterize `model` and maybe `request_id`.  
   - `harmony_state_to_openai_final(parser, model)` (non-streaming).

3. Wire them into:
   - `iter_sse()` in `handle_chat_with_harmony` for streaming clients.  
   - The non-streaming path after collecting the whole upstream response.  

4. Optionally, add tests using a small synthetic Harmony output (from the Harmony docs examples) to verify that:
   - Tool calls produce the expected `tool_calls` structures.[1]
   - Final channel messages end up as assistant `content`.  

If you want, we can next iterate on specific synthetic Harmony transcripts and write test cases the agent can use to verify the mapping end-to-end.

Sources
[1] OpenAI Harmony Response Format https://developers.openai.com/cookbook/articles/openai-harmony/
[2] openai/harmony - GitHub https://github.com/openai/harmony
[3] Have you noticed weird tool-calling behaviour from GPT-OSS when ... https://www.linkedin.com/posts/tanmayb123_have-you-noticed-weird-tool-calling-behaviour-activity-7377914599469277184-Ms-9
[4] harmony_utils - vLLM https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/parser/harmony_utils/
[5] Build a Weather Assistant with OpenAI GPT-OSS and Harmony SDK ... https://vast.ai/article/build-a-weather-assistant-with-openai-gpt-oss-and-harmony-sdk-on-vast-ai
[6] serving_chat_stream_harmony - vLLM https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/serving_chat_stream_harmony/
[7] Function calling | OpenAI API https://platform.openai.com/docs/guides/function-calling
[8] Structured output with OpenAI SDK and gpt-oss:20b not working https://github.com/ollama/ollama/issues/11691?timeline_page=1
[9] harmony package - github.com/euforicio/harmony-go - Go Packages https://pkg.go.dev/github.com/euforicio/harmony-go
[10] [Usage]: gpt-oss-120b tool calls · Issue #22337 - GitHub https://github.com/vllm-project/vllm/issues/22337
[11] Running OpenAI's GPT-OSS on Vast.ai https://vast.ai/article/running-gpt-oss-on-vast
[12] How is everyone dealing with the OpenAI Harmony format on gpt-oss? https://www.reddit.com/r/LocalLLaMA/comments/1mj6y6j/how_is_everyone_dealing_with_the_openai_harmony/
[13] openai/gpt-oss-120b · Errors in chat template compared to spec https://huggingface.co/openai/gpt-oss-120b/discussions/69
[14] Deploying GPT-OSS models on Vast.ai with vLLM and harmony ... https://www.linkedin.com/posts/vast-ai_running-openais-gpt-oss-on-vastai-activity-7364811204025806849-e511
[15] Streaming using Structured Outputs - OpenAI Developer Community https://community.openai.com/t/streaming-using-structured-outputs/925799



===> PLease, continue.


You now need testable mapping logic + examples so the coding agent can verify the Harmony proxy is correct.

***

## 1. Synthetic Harmony → OpenAI test cases

These are based on the official Harmony function-calling examples.[1][2][3]

### 1.1 Single tool call + final answer

Harmony output snippet (simplified from weather example):[3][1]

```text
<|start|>assistant<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>
<|start|>functions.get_current_weather to=assistant<|channel|>commentary<|message|>{"sunny": true, "temperature": 20}<|end|>
<|start|>assistant<|channel|>final<|message|>It's 20°C and sunny in San Francisco.<|return|>
```



**Expected behavior:**

- `analysis` is ignored.  
- `assistant commentary to=functions.get_current_weather` yields a **tool_call** for `get_current_weather` with arguments `{"location":"San Francisco"}`.  
- Tool result (`functions.get_current_weather to=assistant`) is not surfaced to the user; it should be passed back into a *second* GPT‑OSS call as a tool message (that’s outside this proxy’s scope).[4][1]
- `assistant final` becomes the user-visible answer.

**Streaming deltas:**

1. One delta with:

```json
{
  "choices": [
    {
      "delta": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_x",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\":\"San Francisco\"}"
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}
```

2. Then content deltas for `"It's 20°C and sunny in San Francisco."` on `final` channel.[5][1]

**Non-streaming:**

- `choices[0].message.tool_calls[0].function.name == "get_current_weather"`.  
- `choices[0].message.tool_calls[0].function.arguments == "{\"location\":\"San Francisco\"}"`.  
- `choices[0].message.content == "It's 20°C and sunny in San Francisco."` (after the follow-up call that includes tool result).  

The coding agent can create a unit test that feeds this Harmony text through `StreamableParser` and asserts the resulting OpenAI JSON matches.

***

## 2. Implementing `harmony_state_to_openai_deltas`

Enhance the earlier sketch with minimal state to avoid emitting multiple identical tool_call chunks for a single call. Harmony uses `<|call|>` to mark call boundaries; `StreamableParser` effectively knows when the tool call message is done. [1]

You can add:

```python
class HarmonySessionState:
    def __init__(self):
        self.emitted_tool_call_for_message = False
```

Then:

```python
def harmony_state_to_openai_deltas(parser, model: str, state: HarmonySessionState):
    deltas = []

    role = parser.current_role
    channel = parser.current_channel
    recipient = parser.current_recipient
    delta_text = parser.last_content_delta or ""
    content = parser.current_content or ""

    # Drop analysis
    if channel == "analysis":
        return deltas

    # Tool call: only emit once per message
    if (channel == "commentary"
        and recipient
        and recipient.startswith("functions.")
        and not state.emitted_tool_call_for_message):
        fn_name = recipient.split(".", 1)[1]
        args_text = content  # full JSON
        state.emitted_tool_call_for_message = True

        delta = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"call_{uuid.uuid4().hex}",
                                "type": "function",
                                "function": {
                                    "name": fn_name,
                                    "arguments": args_text,
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        deltas.append(delta)
        return deltas

    # Final visible content
    if channel == "final" and role == "assistant" and delta_text:
        delta = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": delta_text,
                    },
                    "finish_reason": None,
                }
            ],
        }
        deltas.append(delta)
        return deltas

    return deltas
```

Inside `iter_sse()` you’d create a `HarmonySessionState()` once and pass it into each call. When Harmony moves to a new message (e.g., new channel and role), you can reset `state.emitted_tool_call_for_message = False`. The parser can expose message boundaries; if not, you can detect a channel/recipient change.[6][1]

***

## 3. Implementing `harmony_state_to_openai_final`

For non-streaming, better to accumulate per-channel content explicitly rather than just using the last message:

```python
class HarmonyAccumulated:
    def __init__(self):
        self.final_content = []
        self.tool_name = None
        self.tool_args = None

    def add_from_parser(self, parser):
        role = parser.current_role
        channel = parser.current_channel
        recipient = parser.current_recipient
        content = parser.current_content or ""

        if channel == "final" and role == "assistant":
            self.final_content.append(content)
        elif channel == "commentary" and recipient and recipient.startswith("functions."):
            self.tool_name = recipient.split(".", 1)[1]
            self.tool_args = content
```

Then after processing all text:

```python
def harmony_state_to_openai_final(acc: HarmonyAccumulated, model: str):
    message = {
        "role": "assistant",
        "content": "".join(acc.final_content) if acc.final_content else None,
    }
    finish_reason = "stop"

    if acc.tool_name:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": acc.tool_name,
                    "arguments": acc.tool_args,
                },
            }
        ]
        # If there are tool_calls, finish_reason should reflect that
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
```

During parsing:

```python
acc = HarmonyAccumulated()
parser = StreamableParser(ENC, role=Role.ASSISTANT)
parser.process(full_text)
acc.add_from_parser(parser)
completion = harmony_state_to_openai_final(acc, model)
```

If you need multiple intermediate messages from Harmony (e.g., multiple tool calls), you can track a list of `(channel, recipient, content)` snapshots instead of a single aggregate. vLLM’s harmony utilities do something similar to reconstruct all channels.[7][6]

***

## 4. Test strategy for the coding agent

The agent should add tests using synthetic Harmony outputs like those in the OpenAI Harmony article and the ChatML vs Harmony blog.[1][3]

### 4.1 Unit tests

For each synthetic transcript:

1. Create `harmony_text` string.  
2. Feed it into `StreamableParser` chunk by chunk (simulate streaming).  
3. After each `process`, call `harmony_state_to_openai_deltas` and accumulate deltas.  
4. Assert:
   - Tool deltas appear with correct `function.name` and arguments JSON.  
   - Content deltas concatenate to the final answer.  
5. For non-streaming, feed the entire string once, aggregate with `HarmonyAccumulated`, and assert final OpenAI completion matches expectations.

Sources for realistic examples:

- Harmony weather tool examples.[2][1]
- ChatML vs Harmony blog showing tool calls and final answer for Tokyo weather.[3]
- vLLM’s GPT‑OSS + Harmony examples.[6][7]

### 4.2 Integration tests (optional)

Once the proxy is wired to a real llama-swap + GPT‑OSS:

- Use the Harmony SDK to build a conversation+tools, run GPT‑OSS once directly via llama-swap, capture raw Harmony text.[4][6]
- Then replay that text into `StreamableParser` + mapping functions and assert the OpenAI view lines up (tool names, arguments, final answer).

***

This is enough detail for the coding agent to implement and verify the Harmony mapping layer in the proxy without ambiguity.

Sources
[1] OpenAI Harmony Response Format https://developers.openai.com/cookbook/articles/openai-harmony/
[2] openai/harmony - GitHub https://github.com/openai/harmony
[3] ChatML vs Harmony: Understanding the new Format from OpenAI https://huggingface.co/blog/kuotient/chatml-vs-harmony
[4] Verifying gpt-oss implementations - OpenAI Cookbook https://cookbook.openai.com/articles/gpt-oss/verifying-implementations
[5] Function calling | OpenAI API https://platform.openai.com/docs/guides/function-calling
[6] How to run gpt-oss with vLLM - OpenAI for developers https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm/
[7] harmony_utils - vLLM https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/parser/harmony_utils/
[8] Have you noticed weird tool-calling behaviour from GPT-OSS when ... https://www.linkedin.com/posts/tanmayb123_have-you-noticed-weird-tool-calling-behaviour-activity-7377914599469277184-Ms-9
[9] openai/gpt-oss-20b · Tool Calling in Chat Template - Hugging Face https://huggingface.co/openai/gpt-oss-20b/discussions/160
[10] OpenAI Cookbook https://cookbook.openai.com
[11] Build a Weather Assistant with OpenAI GPT-OSS and Harmony SDK ... https://vast.ai/article/build-a-weather-assistant-with-openai-gpt-oss-and-harmony-sdk-on-vast-ai
[12] OpenAI Chat Completion Tool Calls With Reasoning - vLLM https://docs.vllm.ai/en/latest/examples/online_serving/openai_chat_completion_tool_calls_with_reasoning/
[13] Tool results as model input fails with harmony backed models (gpt ... https://github.com/openai/openai-agents-python/issues/1605
[14] Harmony Response Format sometimes outputted when using gpt ... https://forum.langchain.com/t/harmony-response-format-sometimes-outputted-when-using-gpt-oss-120b-as-an-agent/2554
[15] Unit Testing - vLLM https://docs.vllm.ai/en/latest/contributing/model/tests/
[16] a response format designed for GPT-OSS model series : r/LocalLLaMA https://www.reddit.com/r/LocalLLaMA/comments/1midvvn/introduce_openai_harmony_a_response_format/
[17] OpenAI introduces Harmony format for GPT models - LinkedIn https://www.linkedin.com/posts/winklerjenna_ai-agenticai-openai-activity-7361063295761457152-tigo



