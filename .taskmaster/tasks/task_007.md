# Task ID: 7

**Title:** Implement transparent passthrough for non-Harmony endpoints

**Status:** pending

**Dependencies:** 6

**Priority:** medium

**Description:** Create generic passthrough functionality for all non-Harmony models and endpoints

**Details:**

Implement `proxy_openai_endpoint(path, body, stream)` that:
- Forwards requests byte-for-byte to llama-swap
- Preserves all headers and HTTP methods
- Handles streaming responses with `httpx.AsyncClient.stream`
- Maintains response status codes and error handling
- Works for any endpoint path (chat/completions, completions, embeddings, etc.)

Ensure the proxy is completely transparent for non-Harmony models - clients should see identical behavior as if talking directly to llama-swap.

**Test Strategy:**

Test with various OpenAI endpoints, verify headers are preserved, test streaming and non-streaming responses, compare responses with direct llama-swap calls to ensure transparency
