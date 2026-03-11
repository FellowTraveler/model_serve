# Task ID: 3

**Title:** Implement OpenAI to Harmony message conversion

**Status:** pending

**Dependencies:** 2

**Priority:** high

**Description:** Create functions to convert OpenAI message format to Harmony Conversation objects

**Details:**

Implement `openai_messages_to_harmony(messages)` that maps:
- `system` → `Role.SYSTEM` with `SystemContent`
- `user` → `Role.USER`
- `assistant` → `Role.ASSISTANT`
- Handle `tool` messages for future multi-turn support

Implement `openai_tools_to_harmony(tools)` that converts OpenAI tool definitions to `ToolDescription` objects with name, description, and parameters.

Implement `build_conversation(body)` that combines messages and tools into a `Conversation` object.

Implement `render_harmony_prompt(convo)` using `ENC.render_conversation_for_completion(convo, Role.ASSISTANT)`

**Test Strategy:**

Unit tests with sample OpenAI messages and tools, verify correct Harmony objects are created, test edge cases like empty messages or missing tool descriptions
