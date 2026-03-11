# Task ID: 4

**Title:** Implement Harmony to OpenAI response conversion

**Status:** pending

**Dependencies:** 3

**Priority:** high

**Description:** Create functions to parse Harmony output and convert it back to OpenAI-compatible responses

**Details:**

Implement `HarmonySessionState` class to track streaming state and avoid duplicate tool call emissions.

Implement `harmony_state_to_openai_deltas(parser, model, state)` that:
- Ignores `analysis` channel
- Converts `commentary` + `recipient=functions.foo` to OpenAI tool_calls deltas
- Maps `final` channel content to assistant content deltas

Implement `HarmonyAccumulated` class and `harmony_state_to_openai_final(acc, model)` for non-streaming responses that accumulates all channels and produces final OpenAI completion response with proper `finish_reason` ("stop" or "tool_calls")

**Test Strategy:**

Unit tests with synthetic Harmony output strings, verify tool calls are extracted correctly, test that analysis channel is dropped, verify streaming deltas match expected OpenAI format
