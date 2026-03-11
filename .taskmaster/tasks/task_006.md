# Task ID: 6

**Title:** Implement OpenAI chat completions endpoint with Harmony support

**Status:** pending

**Dependencies:** 5

**Priority:** high

**Description:** Create the `/v1/chat/completions` endpoint that handles both Harmony and passthrough modes with streaming support

**Details:**

Implement `@app.post("/v1/chat/completions")` that:
- Reads request body and extracts model and stream flag
- Routes to `handle_chat_with_harmony()` for Harmony models
- Routes to `proxy_openai_endpoint()` for other models

Implement `handle_chat_with_harmony()` that:
- Converts OpenAI request to Harmony format
- Calls llama-swap with Harmony prompt
- Streams response through `StreamableParser`
- Converts back to OpenAI format using the conversion functions
- Supports both streaming (SSE) and non-streaming responses

**Test Strategy:**

Integration tests with mock llama-swap responses, test both streaming and non-streaming modes, verify proper SSE formatting, test error handling for malformed requests
