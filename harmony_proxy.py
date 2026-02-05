"""
Harmony Proxy - OpenAI/Anthropic compatible proxy with Harmony support for GPT-OSS models.

This proxy sits in front of llama-swap and:
- Applies Harmony encoding/decoding for GPT-OSS models
- Passes through all other models unchanged
- Supports both OpenAI and Anthropic APIs
"""

import os
import json
import uuid
import logging
import yaml
import httpx
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("harmony_proxy")

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    Message,
    Author,
    Role,
    TextContent,
    SystemContent,
    DeveloperContent,
    ToolDescription,
    ToolNamespaceConfig,
    StreamableParser,
)

# Configuration
# Construct LLAMA_SWAP_BASE from port, consistent with .env
LLAMA_SWAP_PORT = os.environ.get("LLAMA_SWAP_PORT", "5847")
LLAMA_SWAP_BASE = os.environ.get("LLAMA_SWAP_BASE", f"http://127.0.0.1:{LLAMA_SWAP_PORT}")

# Ollama backend for MXFP4 models (llama.cpp doesn't support MXFP4)
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")


def is_mxfp4_model(model: str) -> bool:
    """Check if model requires MXFP4 quantization (route to Ollama)."""
    return "mxfp4" in model.lower()

# Load Harmony models list from config
# Model names must match llama-swap config.yaml exactly
def load_harmony_models():
    default_config = os.path.join(os.path.dirname(__file__), "harmony_models.yaml")
    config_path = os.environ.get("HARMONY_MODELS_CONFIG", default_config)
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        models = set(cfg.get("harmony_models", []))
        logger.info(f"Loaded {len(models)} Harmony models from config")
        return models
    except FileNotFoundError:
        logger.warning(f"{config_path} not found, no models will use Harmony")
        return set()
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in {config_path}: {e}")
        return set()

HARMONY_MODELS = load_harmony_models()

# Load GPT-OSS Harmony encoding once at startup
ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# FastAPI app
app = FastAPI(title="Harmony Proxy", description="OpenAI/Anthropic proxy with Harmony support")

# Request logging directory
REQUEST_LOG_DIR = Path(os.path.dirname(__file__)) / "request_logs"
REQUEST_LOG_DIR.mkdir(exist_ok=True)


def log_request(endpoint: str, body: dict, source: str = "unknown"):
    """Log incoming request to file for debugging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model = body.get("model", "unknown")
    log_file = REQUEST_LOG_DIR / f"{timestamp}_{model.replace('/', '_')}_{endpoint.replace('/', '_')}.json"

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "source": source,
        "body": body,
    }

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Logged request to {log_file}")
    return log_file


def is_harmony_model(model: str) -> bool:
    """Check if model requires Harmony encoding.

    This includes:
    - Models explicitly listed in HARMONY_MODELS
    - MXFP4 models (GPT-OSS quantized with MXFP4, routed to Ollama)
    """
    return model in HARMONY_MODELS or is_mxfp4_model(model)


# ============================================================================
# OpenAI to Harmony Conversion Functions
# ============================================================================

def openai_messages_to_harmony(messages: list) -> list:
    """
    Convert OpenAI message format to Harmony Message objects.

    Role mapping:
    - system -> Role.SYSTEM
    - user -> Role.USER
    - assistant -> Role.ASSISTANT (with tool_calls -> commentary channel)
    - developer -> Role.DEVELOPER
    - tool -> Role.TOOL (with tool name from tool_call_id mapping)
    """
    harmony_messages = []

    # Track tool_call_id -> tool_name mapping for tool results
    tool_call_id_to_name = {}

    for m in messages:
        role_str = m.get("role", "")
        content = m.get("content", "")
        tool_calls = m.get("tool_calls", [])
        tool_call_id = m.get("tool_call_id")

        # Handle assistant messages with tool calls
        if role_str == "assistant" and tool_calls:
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                fn_name = func.get("name", "")
                fn_args = func.get("arguments", "{}")

                # Track mapping for later tool results
                tool_call_id_to_name[tc_id] = fn_name

                # Create Harmony message for tool call
                harmony_messages.append(Message(
                    author=Author(role=Role.ASSISTANT),
                    channel="commentary",
                    recipient=f"functions.{fn_name}",
                    content=[TextContent(text=fn_args)],
                ))
            continue

        # Handle tool result messages
        if role_str == "tool":
            # Get tool name from the mapping
            tool_name = tool_call_id_to_name.get(tool_call_id, "unknown_tool")
            harmony_messages.append(Message(
                author=Author(role=Role.TOOL, name=tool_name),
                content=[TextContent(text=content)] if content else [],
            ))
            continue

        # Map standard OpenAI role string to Harmony Role enum
        role_map = {
            "system": Role.SYSTEM,
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
            "developer": Role.DEVELOPER,
        }

        role = role_map.get(role_str)
        if role is None:
            logger.warning(f"Unknown role '{role_str}' in message, skipping")
            continue

        harmony_messages.append(Message(
            author=Author(role=role),
            content=[TextContent(text=content)] if content else [],
        ))

    return harmony_messages


def openai_tools_to_harmony(tools: list) -> list:
    """
    Convert OpenAI tool definitions to Harmony ToolDescription objects.

    OpenAI format:
    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    harmony_tools = []
    if not tools:
        return harmony_tools

    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        harmony_tools.append(ToolDescription(
            name=fn.get("name", ""),
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {}),
        ))

    return harmony_tools


def build_conversation(body: dict) -> Conversation:
    """
    Build a Harmony Conversation from an OpenAI request body.

    Tools are included via SystemContent in a system message with the
    "functions" namespace, following Harmony convention.
    """
    openai_messages = body.get("messages", [])
    tools = openai_tools_to_harmony(body.get("tools", []))

    # First, convert all messages using openai_messages_to_harmony
    # This properly handles tool calls and tool results
    harmony_messages = openai_messages_to_harmony(openai_messages)

    # If we have tools, we need to replace the system message with one that includes tool definitions
    if tools:
        tool_namespace = ToolNamespaceConfig(
            name="functions",
            description="Available functions",
            tools=tools,
        )

        # Find existing system message and replace it, or add new one
        system_idx = None
        system_text = ""
        for i, msg in enumerate(harmony_messages):
            if msg.author.role == Role.SYSTEM:
                system_idx = i
                # Extract text from existing system content
                for c in msg.content:
                    if isinstance(c, TextContent):
                        system_text = c.text
                        break
                break

        # Create system message with tools
        system_content = SystemContent(
            model_identity=system_text,
            tools={"functions": tool_namespace},
        )
        system_message = Message(
            author=Author(role=Role.SYSTEM),
            content=[system_content],
        )

        if system_idx is not None:
            harmony_messages[system_idx] = system_message
        else:
            harmony_messages.insert(0, system_message)

    return Conversation(messages=harmony_messages)


def render_harmony_prompt(convo: Conversation) -> str:
    """
    Render a Harmony Conversation into a prompt string for GPT-OSS.
    Uses the GPT-OSS encoding to prepare for assistant completion.
    """
    tokens = ENC.render_conversation_for_completion(convo, Role.ASSISTANT)
    return ENC.decode(tokens)


# ============================================================================
# Harmony to OpenAI Conversion Functions
# ============================================================================

class HarmonySessionState:
    """
    Track streaming state to avoid duplicate emissions.
    Reset when channel/recipient changes (new message).
    """
    def __init__(self):
        self.emitted_tool_call_for_message = False
        self.emitted_role = False  # Only emit role in first content delta
        self.current_tool_call_id = None
        self.last_channel = None
        self.last_recipient = None
        self.has_tool_calls = False  # Track if ANY tool calls were made (for finish_reason)
        self.has_real_content = False  # Track if we emitted commentary tool calls or final content
        self.deferred_analysis_tool_calls = []  # Buffer analysis channel tool calls as fallback

    def check_and_update(self, channel: str, recipient: str) -> bool:
        """Check if state changed and update. Returns True if new message started."""
        if channel != self.last_channel or recipient != self.last_recipient:
            self.last_channel = channel
            self.last_recipient = recipient
            self.emitted_tool_call_for_message = False
            self.emitted_role = False
            self.current_tool_call_id = None
            return True
        return False


def harmony_state_to_openai_deltas(parser: StreamableParser, model: str, state: HarmonySessionState, chunk_id: str, created: int) -> list:
    """
    Convert StreamableParser state to OpenAI streaming deltas.

    Channel handling:
    - commentary + functions.X recipient: TOOL CALL (preferred)
    - analysis + functions.X recipient: DEFERRED tool call (only used if no real content)
    - analysis (without functions recipient): IGNORE (internal reasoning)
    - commentary (without functions recipient): IGNORE
    - final: USER-VISIBLE content
    """
    deltas = []

    channel = parser.current_channel
    recipient = parser.current_recipient
    delta_text = parser.last_content_delta or ""

    # Track state changes
    state.check_and_update(channel, recipient)

    # Tool call with functions.X recipient
    if recipient and recipient.startswith("functions."):
        fn_name = recipient.split(".", 1)[1]

        # Analysis channel tool calls are deferred (likely hallucinated chain-of-thought)
        # Only emit them at the end if there's no other content
        if channel == "analysis":
            # Buffer the tool call info for potential later use
            if not state.emitted_tool_call_for_message:
                state.emitted_tool_call_for_message = True
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                state.current_tool_call_id = tool_call_id
                state.deferred_analysis_tool_calls.append({
                    "fn_name": fn_name,
                    "tool_call_id": tool_call_id,
                    "arguments": delta_text,
                })
            elif delta_text and state.deferred_analysis_tool_calls:
                # Append to the last deferred tool call's arguments
                state.deferred_analysis_tool_calls[-1]["arguments"] += delta_text
            return deltas  # Don't emit yet

        # Commentary channel tool calls are emitted immediately (preferred)
        if not state.emitted_tool_call_for_message:
            # First chunk: emit tool call header with function name
            state.emitted_tool_call_for_message = True
            state.has_tool_calls = True  # Track for finish_reason
            state.has_real_content = True  # Mark that we have real content
            state.current_tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

            delta = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": state.current_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": "",
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            }
            deltas.append(delta)

        # Stream arguments incrementally
        if delta_text:
            delta = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": delta_text,
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            }
            deltas.append(delta)

        return deltas

    # Final channel: user-visible assistant content
    if channel == "final" and delta_text:
        state.has_real_content = True  # Mark that we have real content

        # Build delta - only include role in first chunk (OpenAI spec)
        delta_content = {"content": delta_text}
        if not state.emitted_role:
            delta_content["role"] = "assistant"
            state.emitted_role = True

        delta = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta_content,
                "finish_reason": None,
            }],
        }
        deltas.append(delta)
        return deltas

    return deltas


class HarmonyAccumulated:
    """
    Accumulate Harmony parser output for non-streaming responses.
    """
    def __init__(self):
        self.final_content = []
        self.tool_calls = []  # List of (name, args) tuples - preferred (from commentary channel)
        self.analysis_tool_calls = []  # Fallback tool calls from analysis channel

    def add_from_parser(self, parser: StreamableParser):
        """Extract and accumulate data from parser's completed messages."""
        for msg in parser.messages:
            channel = msg.channel
            recipient = msg.recipient

            # Get text content
            text = ""
            for c in msg.content:
                if hasattr(c, "text"):
                    text = c.text
                    break

            if channel == "final":
                self.final_content.append(text)
            elif recipient and recipient.startswith("functions."):
                fn_name = recipient.split(".", 1)[1]
                if channel == "analysis":
                    # Analysis channel tool calls are fallback (likely hallucinated)
                    self.analysis_tool_calls.append((fn_name, text))
                else:
                    # Commentary channel tool calls are preferred
                    self.tool_calls.append((fn_name, text))

    def get_effective_tool_calls(self):
        """Get tool calls to use: prefer commentary, fall back to analysis if no other content."""
        if self.tool_calls:
            return self.tool_calls
        if self.final_content:
            return []  # Have final content, don't use analysis tool calls
        # No real content, use analysis tool calls as fallback
        return self.analysis_tool_calls


def harmony_state_to_openai_final(acc: HarmonyAccumulated, model: str) -> dict:
    """
    Build a non-streaming OpenAI chat completion from accumulated Harmony data.
    """
    import time

    message = {
        "role": "assistant",
        "content": "".join(acc.final_content) if acc.final_content else None,
    }
    finish_reason = "stop"

    # Use effective tool calls (prefers commentary, falls back to analysis if no other content)
    effective_tool_calls = acc.get_effective_tool_calls()
    if effective_tool_calls:
        message["tool_calls"] = []
        for fn_name, args in effective_tool_calls:
            message["tool_calls"].append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": args,
                },
            })
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "harmony_models": list(HARMONY_MODELS),
        "llama_swap_base": LLAMA_SWAP_BASE,
    }


# ============================================================================
# HTTP Endpoints
# ============================================================================

async def proxy_openai_endpoint(path: str, body: dict, stream: bool):
    """
    Transparent passthrough to llama-swap for non-Harmony models.
    Preserves request/response byte-for-byte.
    """
    url = f"{LLAMA_SWAP_BASE}{path}"

    if stream:
        # Client must be created inside the generator to avoid closing before iteration
        async def iter_stream():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            model = body.get("model", "unknown").replace("/", "_")
            response_log = REQUEST_LOG_DIR / f"{timestamp}_{model}_response.jsonl"

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=body, timeout=None) as resp:
                    with open(response_log, "w") as f:
                        async for chunk in resp.aiter_raw():
                            # Log raw chunk
                            f.write(chunk.decode("utf-8", errors="replace"))
                            yield chunk
                    logger.info(f"Logged streaming response to {response_log}")
        return StreamingResponse(iter_stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(url, json=body, timeout=None)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def handle_chat_with_harmony(body: dict, stream: bool):
    """
    Handle chat completions for Harmony models (GPT-OSS).
    Converts OpenAI format to Harmony, calls backend, converts back.

    Routes to:
    - Ollama for MXFP4 models (llama.cpp doesn't support MXFP4)
    - llama-swap for all other Harmony models
    """
    model = body.get("model", "")
    use_ollama = is_mxfp4_model(model)

    # Convert OpenAI request to Harmony format
    convo = build_conversation(body)
    harmony_prompt = render_harmony_prompt(convo)

    # Debug: log the Harmony prompt (first 500 chars)
    logger.debug(f"Harmony prompt (first 500 chars): {harmony_prompt[:500]}")

    if use_ollama:
        # Ollama backend - use /api/generate with raw mode
        # CRITICAL: Override stop sequences - Ollama's default stops at Harmony tokens
        # which breaks streaming. We need full Harmony output for parsing.
        backend_req = {
            "model": model,
            "prompt": harmony_prompt,
            "stream": True,
            "raw": True,  # Disable Ollama's chat templating
            "options": {
                "stop": [],  # Clear default Harmony stop tokens so we get full output
            },
        }
        # Ollama uses different parameter names
        if "max_tokens" in body:
            backend_req["options"]["num_predict"] = body["max_tokens"]
        if "temperature" in body:
            backend_req["options"]["temperature"] = body["temperature"]
        url = f"{OLLAMA_BASE}/api/generate"
        logger.info(f"Routing MXFP4 model to Ollama: {url}")
    else:
        # llama-swap backend - use /v1/completions
        backend_req = {
            "model": model,
            "prompt": harmony_prompt,
            "stream": True,  # Always stream from upstream
        }
        # Copy through other parameters
        for key in ("max_tokens", "temperature", "top_p", "stop"):
            if key in body:
                backend_req[key] = body[key]
        url = f"{LLAMA_SWAP_BASE}/v1/completions"

    if stream:
        # Streaming response - client must be created inside generator
        # to avoid closing before iteration starts
        async def iter_sse():
            import time
            parser = StreamableParser(ENC, role=Role.ASSISTANT)
            state = HarmonySessionState()
            harmony_parse_failed = False
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            chunks_received = 0
            deltas_emitted = 0
            accumulated_raw = []  # For debugging if nothing is emitted

            logger.info(f"Starting streaming request to {url}")
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=backend_req, timeout=None) as resp:
                    logger.info(f"Upstream response status: {resp.status_code}")

                    # Handle upstream errors (e.g., context length exceeded)
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        error_text = error_body.decode("utf-8", errors="replace")
                        logger.error(f"Upstream error {resp.status_code}: {error_text[:500]}")

                        # Return error as assistant message so Goose knows what happened
                        error_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": f"Error from model server: {error_text[:200]}"},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        stop_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(stop_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue

                        # Parse line based on backend format
                        if use_ollama:
                            # Ollama: NDJSON format {"response": "...", "done": false}
                            try:
                                chunk = json.loads(line)
                                if chunk.get("done"):
                                    break
                                content = chunk.get("response", "")
                            except json.JSONDecodeError:
                                continue
                        else:
                            # llama-swap: SSE format "data: {...}"
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                content = chunk.get("choices", [{}])[0].get("text", "")
                            except json.JSONDecodeError:
                                continue

                        chunks_received += 1
                        if content:
                            accumulated_raw.append(content)
                            if harmony_parse_failed:
                                # After parse failure, stream raw content
                                raw_chunk = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": content}}],
                                }
                                yield f"data: {json.dumps(raw_chunk)}\n\n"
                            else:
                                try:
                                    tokens = ENC.encode(content, allowed_special='all')
                                    for token in tokens:
                                        parser.process(token)
                                        deltas = harmony_state_to_openai_deltas(parser, model, state, chunk_id, created)
                                        for delta in deltas:
                                            deltas_emitted += 1
                                            yield f"data: {json.dumps(delta)}\n\n"
                                except Exception as e:
                                    logger.warning(f"Harmony parse error in stream, falling back to raw: {e}")
                                    logger.debug(f"Raw content that failed Harmony parse: {content[:200]}")
                                    harmony_parse_failed = True
                                    # Yield current content as raw
                                    raw_chunk = {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "model": model,
                                        "choices": [{"index": 0, "delta": {"content": content}}],
                                    }
                                    yield f"data: {json.dumps(raw_chunk)}\n\n"

            # Finalize parsing by sending <|end|> if we're still in Harmony mode
            if not harmony_parse_failed:
                try:
                    tokens = ENC.encode("<|end|>", allowed_special='all')
                    for token in tokens:
                        parser.process(token)
                        deltas = harmony_state_to_openai_deltas(parser, model, state, chunk_id, created)
                        for delta in deltas:
                            yield f"data: {json.dumps(delta)}\n\n"
                except Exception:
                    pass  # Ignore finalization errors in streaming

            # Emit deferred analysis tool calls if we have no other content
            # These are likely hallucinated chain-of-thought, but better than nothing
            if not state.has_real_content and state.deferred_analysis_tool_calls:
                logger.info(f"Emitting {len(state.deferred_analysis_tool_calls)} deferred analysis tool call(s) as fallback")
                for deferred in state.deferred_analysis_tool_calls:
                    # Emit tool call header
                    header_delta = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [{
                                    "index": 0,
                                    "id": deferred["tool_call_id"],
                                    "type": "function",
                                    "function": {
                                        "name": deferred["fn_name"],
                                        "arguments": "",
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(header_delta)}\n\n"
                    deltas_emitted += 1

                    # Emit arguments
                    if deferred["arguments"]:
                        args_delta = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "function": {
                                            "arguments": deferred["arguments"],
                                        },
                                    }],
                                },
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(args_delta)}\n\n"
                        deltas_emitted += 1

                    state.has_tool_calls = True

            # Log parser state if we emitted nothing (helps debug)
            if deltas_emitted == 0:
                raw_text = "".join(accumulated_raw)
                logger.warning(f"No deltas emitted! Raw content ({len(raw_text)} chars): {raw_text[:500]}")
                logger.warning(f"Parser state: messages={len(parser.messages)}, "
                             f"current_channel={parser.current_channel}, current_recipient={parser.current_recipient}")
                for i, msg in enumerate(parser.messages):
                    logger.warning(f"  Message {i}: channel={msg.channel}, recipient={msg.recipient}")

            # Emit final chunk with appropriate finish_reason
            logger.info(f"Stream complete: chunks_received={chunks_received}, deltas_emitted={deltas_emitted}, has_tool_calls={state.has_tool_calls}")
            finish_reason = "tool_calls" if state.has_tool_calls else "stop"
            stop_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(iter_sse(), media_type="text/event-stream")
    else:
        # Non-streaming response
        parser = StreamableParser(ENC, role=Role.ASSISTANT)
        raw_content = []  # Fallback if Harmony parsing fails
        harmony_parse_failed = False

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=backend_req, timeout=None) as resp:
                # Handle upstream errors (e.g., context length exceeded)
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    logger.error(f"Upstream error {resp.status_code}: {error_text[:500]}")

                    return JSONResponse(
                        content={
                            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "object": "chat.completion",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": f"Error from model server: {error_text[:200]}"},
                                "finish_reason": "stop",
                            }],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        },
                        status_code=resp.status_code,
                    )

                async for line in resp.aiter_lines():
                    if not line:
                        continue

                    # Parse line based on backend format
                    if use_ollama:
                        # Ollama: NDJSON format {"response": "...", "done": false}
                        try:
                            chunk = json.loads(line)
                            if chunk.get("done"):
                                break
                            content = chunk.get("response", "")
                        except json.JSONDecodeError:
                            continue
                    else:
                        # llama-swap: SSE format "data: {...}"
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("text", "")
                        except json.JSONDecodeError:
                            continue

                    if content:
                        raw_content.append(content)
                        if not harmony_parse_failed:
                            try:
                                tokens = ENC.encode(content, allowed_special='all')
                                for token in tokens:
                                    parser.process(token)
                            except Exception as e:
                                logger.warning(f"Harmony parse error, falling back to raw: {e}")
                                harmony_parse_failed = True

        if harmony_parse_failed:
            # Return raw content as plain response
            return JSONResponse(content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(raw_content)},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        # Finalize parsing by adding <|end|> if content doesn't end with it
        raw_text = "".join(raw_content)
        if raw_text and not raw_text.rstrip().endswith("<|end|>"):
            try:
                tokens = ENC.encode("<|end|>", allowed_special='all')
                for token in tokens:
                    parser.process(token)
            except Exception as e:
                logger.warning(f"Failed to finalize Harmony parsing: {e}")

        # Log what we actually received and parsed
        logger.info(f"Raw content received: {len(raw_text)} chars, first 200: {raw_text[:200]}")
        logger.info(f"Parser messages count: {len(parser.messages)}")
        for i, msg in enumerate(parser.messages):
            logger.info(f"Parser message {i}: channel={getattr(msg, 'channel', None)}, recipient={getattr(msg, 'recipient', None)}")

        # Accumulate and return final response
        acc = HarmonyAccumulated()
        acc.add_from_parser(parser)
        logger.info(f"Accumulated: final_content={len(acc.final_content)}, tool_calls={len(acc.tool_calls)}")
        return JSONResponse(content=harmony_state_to_openai_final(acc, model))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    NOTE: llama-swap/llama.cpp handles Harmony encoding/decoding via Jinja templates,
    so we pass through all models transparently. The proxy still serves as a unified
    endpoint and can add logging/filtering in the future.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model", "")
    stream = body.get("stream", False)

    # Log request for debugging
    user_agent = request.headers.get("user-agent", "unknown")
    log_request("/v1/chat/completions", body, source=user_agent)

    logger.info(f"Chat completion request: model={model}, stream={stream}, harmony={is_harmony_model(model)}")

    # Route based on model type
    if is_harmony_model(model):
        # GPT-OSS models: use Harmony encoding/decoding (llama-server runs with --no-jinja)
        return await handle_chat_with_harmony(body, stream)
    else:
        # Other models: passthrough to llama-swap
        return await proxy_openai_endpoint("/v1/chat/completions", body, stream)


# ============================================================================
# Passthrough Endpoints (non-Harmony)
# ============================================================================

@app.post("/v1/completions")
async def completions(request: Request):
    """Legacy completions endpoint - passthrough only."""
    body = await request.json()
    stream = body.get("stream", False)
    return await proxy_openai_endpoint("/v1/completions", body, stream)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Embeddings endpoint - passthrough only."""
    body = await request.json()
    return await proxy_openai_endpoint("/v1/embeddings", body, stream=False)


@app.get("/v1/models")
async def list_models():
    """List available models - passthrough to llama-swap."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{LLAMA_SWAP_BASE}/v1/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ============================================================================
# Anthropic Messages API (passthrough only)
# ============================================================================

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """
    Anthropic Messages API - always passthrough, no Harmony conversion.

    Harmony is designed for OpenAI message format, not Anthropic's format.
    Mixing them would risk undefined behavior.

    Preserves Anthropic-specific SSE event types:
    - message_start, content_block_delta, content_block_stop
    - message_delta, message_stop
    """
    body = await request.json()
    stream = body.get("stream", False)

    logger.info(f"Anthropic messages request: model={body.get('model', '')}, stream={stream}")

    url = f"{LLAMA_SWAP_BASE}/v1/messages"
    async with httpx.AsyncClient(timeout=None) as client:
        if stream:
            # Stream raw bytes to preserve Anthropic SSE format
            async def iter_stream():
                async with client.stream("POST", url, json=body, timeout=None) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk
            return StreamingResponse(iter_stream(), media_type="text/event-stream")
        else:
            resp = await client.post(url, json=body, timeout=None)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
