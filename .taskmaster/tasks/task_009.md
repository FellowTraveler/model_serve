# Task ID: 9

**Title:** Add comprehensive test suite with synthetic Harmony examples

**Status:** pending

**Dependencies:** 8

**Priority:** medium

**Description:** Create unit and integration tests using realistic Harmony examples to verify correct parsing and conversion

**Details:**

Create test cases based on official Harmony examples:
- Single tool call + final answer scenario
- Multiple tool calls
- Analysis channel filtering (should be dropped)
- Edge cases like malformed JSON in tool arguments

Test synthetic Harmony output like:
```
<|start|>assistant<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>
<|start|>assistant<|channel|>final<|message|>It's 20°C and sunny in San Francisco.<|return|>
```

Verify expected OpenAI tool_calls and content are generated correctly.

**Test Strategy:**

Automated test suite with pytest, mock llama-swap responses, test both streaming and non-streaming conversion, verify tool call extraction accuracy, test error handling
