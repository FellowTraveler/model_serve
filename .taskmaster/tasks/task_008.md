# Task ID: 8

**Title:** Implement Anthropic Messages API passthrough support

**Status:** pending

**Dependencies:** 7

**Priority:** medium

**Description:** Add support for Anthropic's `/v1/messages` endpoint with complete passthrough to llama-swap

**Details:**

Implement `@app.post("/v1/messages")` endpoint that:
- Always uses passthrough mode (no Harmony conversion for Anthropic format)
- Preserves Anthropic-specific SSE event types (`message_start`, `content_block_delta`, etc.)
- Handles Anthropic's different streaming format
- Maintains compatibility with Claude Code and other Anthropic clients

Rationale: Harmony is designed for OpenAI message format, not Anthropic's format. Mixing them would risk undefined behavior.

**Test Strategy:**

Test with Anthropic-compatible clients, verify SSE event types are preserved, test tool calling through Anthropic format, ensure streaming works correctly
