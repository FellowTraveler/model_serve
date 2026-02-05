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
LLAMA_SWAP_BASE = os.environ.get("LLAMA_SWAP_BASE", "http://127.0.0.1:8000")

# Load Harmony models list from config
# Model names must match llama-swap config.yaml exactly
def load_harmony_models():
    config_path = os.path.join(os.path.dirname(__file__), "harmony_models.yaml")
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


def is_harmony_model(model: str) -> bool:
    """Check if model requires Harmony encoding (O(1) lookup)."""
    return model in HARMONY_MODELS


# ============================================================================
# OpenAI to Harmony Conversion Functions
# ============================================================================

def openai_messages_to_harmony(messages: list) -> list:
    """
    Convert OpenAI message format to Harmony Message objects.

    Role mapping:
    - system -> Role.SYSTEM
    - user -> Role.USER
    - assistant -> Role.ASSISTANT
    - developer -> Role.DEVELOPER
    - tool -> handled separately for multi-turn support
    """
    harmony_messages = []
    for m in messages:
        role_str = m.get("role", "")
        content = m.get("content", "")

        # Map OpenAI role string to Harmony Role enum
        role_map = {
            "system": Role.SYSTEM,
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
            "developer": Role.DEVELOPER,
        }

        role = role_map.get(role_str)
        if role is None:
            # Skip unknown roles (e.g., tool messages for now)
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

    harmony_messages = []

    # Check if there's already a system message
    has_system = any(m.get("role") == "system" for m in openai_messages)

    # If we have tools, we need a system message with tool definitions
    if tools:
        tool_namespace = ToolNamespaceConfig(
            name="functions",
            description="Available functions",
            tools=tools,
        )
        system_content = SystemContent(
            model_identity=None,  # Use default or extract from system message
            tools={"functions": tool_namespace},
        )

        # Find and merge with existing system message, or create new one
        for m in openai_messages:
            if m.get("role") == "system":
                # Merge system message content into model_identity
                system_content = SystemContent(
                    model_identity=m.get("content", ""),
                    tools={"functions": tool_namespace},
                )
                harmony_messages.append(Message(
                    author=Author(role=Role.SYSTEM),
                    content=[system_content],
                ))
            elif m.get("role") in ("user", "assistant", "developer"):
                role_map = {
                    "user": Role.USER,
                    "assistant": Role.ASSISTANT,
                    "developer": Role.DEVELOPER,
                }
                content = m.get("content", "")
                harmony_messages.append(Message(
                    author=Author(role=role_map[m["role"]]),
                    content=[TextContent(text=content)] if content else [],
                ))

        # If no system message existed, prepend one with tools
        if not has_system:
            harmony_messages.insert(0, Message(
                author=Author(role=Role.SYSTEM),
                content=[system_content],
            ))
    else:
        # No tools - use simple conversion
        harmony_messages = openai_messages_to_harmony(openai_messages)

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
    Track streaming state to avoid duplicate tool call emissions.
    Reset when channel/recipient changes (new message).
    """
    def __init__(self):
        self.emitted_tool_call_for_message = False
        self.last_channel = None
        self.last_recipient = None

    def check_and_update(self, channel: str, recipient: str) -> bool:
        """Check if state changed and update. Returns True if new message started."""
        if channel != self.last_channel or recipient != self.last_recipient:
            self.last_channel = channel
            self.last_recipient = recipient
            self.emitted_tool_call_for_message = False
            return True
        return False


def harmony_state_to_openai_deltas(parser: StreamableParser, model: str, state: HarmonySessionState) -> list:
    """
    Convert StreamableParser state to OpenAI streaming deltas.

    Channel handling:
    - analysis: IGNORE (internal reasoning, not user-visible)
    - commentary + functions.X recipient: TOOL CALL
    - final: USER-VISIBLE content
    """
    deltas = []

    channel = parser.current_channel
    recipient = parser.current_recipient
    delta_text = parser.last_content_delta or ""
    content = parser.current_content or ""

    # Track state changes
    state.check_and_update(channel, recipient)

    # Drop analysis channel (chain-of-thought)
    if channel == "analysis":
        return deltas

    # Tool call: commentary channel with functions.X recipient
    if channel == "commentary" and recipient and recipient.startswith("functions."):
        if not state.emitted_tool_call_for_message:
            fn_name = recipient.split(".", 1)[1]
            state.emitted_tool_call_for_message = True

            delta = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": content,
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
        delta = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": delta_text,
                },
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
        self.tool_calls = []  # List of (name, args) tuples

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
            elif channel == "commentary" and recipient and recipient.startswith("functions."):
                fn_name = recipient.split(".", 1)[1]
                self.tool_calls.append((fn_name, text))


def harmony_state_to_openai_final(acc: HarmonyAccumulated, model: str) -> dict:
    """
    Build a non-streaming OpenAI chat completion from accumulated Harmony data.
    """
    message = {
        "role": "assistant",
        "content": "".join(acc.final_content) if acc.final_content else None,
    }
    finish_reason = "stop"

    if acc.tool_calls:
        message["tool_calls"] = []
        for fn_name, args in acc.tool_calls:
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
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
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
    async with httpx.AsyncClient(timeout=None) as client:
        if stream:
            async def iter_stream():
                async with client.stream("POST", url, json=body, timeout=None) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk
            return StreamingResponse(iter_stream(), media_type="text/event-stream")
        else:
            resp = await client.post(url, json=body, timeout=None)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def handle_chat_with_harmony(body: dict, stream: bool):
    """
    Handle chat completions for Harmony models (GPT-OSS).
    Converts OpenAI format to Harmony, calls llama-swap, converts back.
    """
    model = body.get("model", "")

    # Convert OpenAI request to Harmony format
    convo = build_conversation(body)
    harmony_prompt = render_harmony_prompt(convo)

    # Build request for llama-swap (send Harmony as user message)
    llama_req = {
        "model": model,
        "messages": [{"role": "user", "content": harmony_prompt}],
        "stream": True,  # Always stream from upstream
    }

    # Copy through other parameters
    for key in ("max_tokens", "temperature", "top_p", "stop"):
        if key in body:
            llama_req[key] = body[key]

    url = f"{LLAMA_SWAP_BASE}/v1/chat/completions"

    if stream:
        # Streaming response - client must be created inside generator
        # to avoid closing before iteration starts
        async def iter_sse():
            parser = StreamableParser(ENC, role=Role.ASSISTANT)
            state = HarmonySessionState()
            harmony_parse_failed = False
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=llama_req, timeout=None) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
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
                                            deltas = harmony_state_to_openai_deltas(parser, model, state)
                                            for delta in deltas:
                                                yield f"data: {json.dumps(delta)}\n\n"
                                    except Exception as e:
                                        logger.warning(f"Harmony parse error in stream, falling back to raw: {e}")
                                        harmony_parse_failed = True
                                        # Yield current content as raw
                                        raw_chunk = {
                                            "id": chunk_id,
                                            "object": "chat.completion.chunk",
                                            "model": model,
                                            "choices": [{"index": 0, "delta": {"content": content}}],
                                        }
                                        yield f"data: {json.dumps(raw_chunk)}\n\n"
                        except json.JSONDecodeError:
                            continue

            # Emit final stop chunk
            stop_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
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
            async with client.stream("POST", url, json=llama_req, timeout=None) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
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
                    except json.JSONDecodeError:
                        continue

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

        # Accumulate and return final response
        acc = HarmonyAccumulated()
        acc.add_from_parser(parser)
        return JSONResponse(content=harmony_state_to_openai_final(acc, model))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Routes to Harmony handler for GPT-OSS models, passthrough for others.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model", "")
    stream = body.get("stream", False)

    logger.info(f"Chat completion request: model={model}, stream={stream}, harmony={is_harmony_model(model)}")

    if is_harmony_model(model):
        return await handle_chat_with_harmony(body, stream)
    else:
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
