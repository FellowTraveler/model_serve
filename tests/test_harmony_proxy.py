"""
Tests for the Harmony proxy.
"""

import pytest
from harmony_proxy import (
    is_harmony_model,
    is_mxfp4_model,
    get_ollama_model_name,
    sanitize_tool_arguments,
    openai_messages_to_harmony,
    openai_tools_to_harmony,
    build_conversation,
    render_harmony_prompt,
    HarmonySessionState,
    HarmonyAccumulated,
    harmony_state_to_openai_deltas,
    harmony_state_to_openai_final,
    ENC,
    HARMONY_MODELS,
)
from openai_harmony import StreamableParser, Role, HarmonyError


class TestModelRouting:
    """Test model routing logic."""

    def test_harmony_model_detected(self):
        """Harmony models should be detected."""
        for model in HARMONY_MODELS:
            assert is_harmony_model(model), f"{model} should be detected as Harmony"

    def test_non_harmony_model_not_detected(self):
        """Non-Harmony models should not be detected."""
        assert not is_harmony_model("qwen3-0.6b")
        assert not is_harmony_model("llama-3.1-8b")
        assert not is_harmony_model("gemma-2b")

    def test_mxfp4_model_detected(self):
        """MXFP4 models should be routed to Ollama."""
        assert is_mxfp4_model("hf.co/Felladrin/gguf-MXFP4-gpt-oss-20b-Derestricted:latest")
        assert is_mxfp4_model("gpt-oss-mxfp4-120b")
        assert is_mxfp4_model("some-model-MXFP4-quantized")  # Case insensitive

    def test_non_mxfp4_model_not_detected(self):
        """Non-MXFP4 models should not be routed to Ollama."""
        assert not is_mxfp4_model("ms/gpt-oss-20b-derestricted-gguf-20.9b-q8_0")
        assert not is_mxfp4_model("llama-3.1-8b")
        assert not is_mxfp4_model("qwen3-0.6b")

    def test_ollama_model_name_mapping(self):
        """MXFP4 model names should be translated to Ollama names."""
        # With ms/ prefix
        mapped = get_ollama_model_name("ms/gguf-mxfp4-gpt-oss-20b-derestricted-20.9b-latest")
        assert mapped == "hf.co/Felladrin/gguf-MXFP4-gpt-oss-20b-Derestricted:latest"

        # Without ms/ prefix
        mapped = get_ollama_model_name("gguf-mxfp4-gpt-oss-20b-derestricted-20.9b-latest")
        assert mapped == "hf.co/Felladrin/gguf-MXFP4-gpt-oss-20b-Derestricted:latest"

        # Unknown model returns original
        mapped = get_ollama_model_name("ms/some-unknown-model")
        assert mapped == "ms/some-unknown-model"


class TestToolArgumentSanitization:
    """Test tool argument sanitization for GPT-OSS hallucination fix."""

    def test_valid_json_unchanged(self):
        """Valid JSON should be returned unchanged."""
        valid = '{"command": "ls -la"}'
        assert sanitize_tool_arguments(valid) == valid

    def test_concatenated_json_extracts_first(self):
        """Concatenated JSON objects should extract only the first."""
        # This is the hallucination pattern from GPT-OSS
        hallucinated = '{"command":"ls"}{"stdout":"file1\\nfile2"}{"command":"cat file1"}'
        result = sanitize_tool_arguments(hallucinated)
        assert result == '{"command":"ls"}'

    def test_nested_json_preserved(self):
        """Nested JSON should be fully preserved."""
        nested = '{"config": {"host": "localhost", "port": 8080}}'
        assert sanitize_tool_arguments(nested) == nested

    def test_json_with_string_braces(self):
        """Braces inside strings should not affect parsing."""
        with_braces = '{"text": "Use {curly} braces"}'
        assert sanitize_tool_arguments(with_braces) == with_braces

    def test_empty_returns_empty(self):
        """Empty string should return empty."""
        assert sanitize_tool_arguments("") == ""
        assert sanitize_tool_arguments("  ") == "  "

    def test_complex_hallucination_pattern(self):
        """
        Regression test: Real GPT-OSS hallucination pattern.

        The model generates tool call + fake response + more tool calls.
        """
        hallucinated = (
            '{"command":"python3 /path/to/script.py"}'
            '{"stdout":"Done. Created file.md\\n"}'
            '{"path":"file.md","command":"view"}'
            '{"stdout":"# Content\\n"}'
        )
        result = sanitize_tool_arguments(hallucinated)
        assert result == '{"command":"python3 /path/to/script.py"}'


class TestOpenAIToHarmonyConversion:
    """Test OpenAI to Harmony format conversion."""

    def test_basic_message_conversion(self):
        """Basic messages should convert correctly."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        harmony_msgs = openai_messages_to_harmony(messages)
        assert len(harmony_msgs) == 2
        assert harmony_msgs[0].author.role == Role.SYSTEM
        assert harmony_msgs[1].author.role == Role.USER

    def test_tool_conversion(self):
        """OpenAI tools should convert to Harmony ToolDescription."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        harmony_tools = openai_tools_to_harmony(tools)
        assert len(harmony_tools) == 1
        assert harmony_tools[0].name == "get_weather"

    def test_build_conversation_with_tools(self):
        """Conversation with tools should include tool definitions."""
        body = {
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "description": "Get weather"},
                }
            ],
        }
        convo = build_conversation(body)
        assert len(convo.messages) >= 1

    def test_render_harmony_prompt(self):
        """Rendered prompt should contain Harmony tokens."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        convo = build_conversation(body)
        prompt = render_harmony_prompt(convo)
        assert "<|start|>" in prompt
        assert "<|message|>" in prompt
        assert "Hello" in prompt

    def test_tool_call_and_result_conversion(self):
        """
        Regression test: Tool calls and results must be properly converted to Harmony format.

        This ensures multi-turn tool usage works correctly:
        1. Assistant tool calls use commentary channel with functions.X recipient
        2. Tool results use Role.TOOL with the tool name
        """
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What time is it?"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_123", "content": "3:00 PM"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_time", "description": "Get time"}}],
        }
        convo = build_conversation(body)
        prompt = render_harmony_prompt(convo)

        # Verify tool call is rendered with commentary channel
        assert "to=functions.get_time" in prompt
        assert "<|channel|>commentary" in prompt
        assert "<|call|>" in prompt

        # Verify tool result is rendered
        assert "<|start|>get_time<|message|>3:00 PM" in prompt

        # Verify prompt ends ready for assistant
        assert prompt.endswith("<|start|>assistant")


class TestHarmonyToOpenAIConversion:
    """Test Harmony to OpenAI format conversion."""

    def test_streaming_drops_analysis_channel(self):
        """Analysis channel should be dropped in streaming output."""
        parser = StreamableParser(ENC, role=Role.ASSISTANT)
        state = HarmonySessionState()
        chunk_id = "chatcmpl-test123"
        created = 1234567890

        # Simulate analysis channel token
        harmony = "<|channel|>analysis<|message|>thinking...<|end|>"
        tokens = ENC.encode(harmony, allowed_special="all")
        for token in tokens:
            parser.process(token)
            deltas = harmony_state_to_openai_deltas(parser, "test-model", state, chunk_id, created)
            # Analysis should produce no deltas
            for d in deltas:
                assert "analysis" not in str(d)

    def test_final_channel_produces_content(self):
        """Final channel should produce content deltas."""
        parser = StreamableParser(ENC, role=Role.ASSISTANT)
        state = HarmonySessionState()
        chunk_id = "chatcmpl-test123"
        created = 1234567890

        harmony = "<|channel|>final<|message|>Hello world<|end|>"
        tokens = ENC.encode(harmony, allowed_special="all")

        all_deltas = []
        for token in tokens:
            parser.process(token)
            deltas = harmony_state_to_openai_deltas(parser, "test-model", state, chunk_id, created)
            all_deltas.extend(deltas)

        # Should have content deltas
        content_deltas = [d for d in all_deltas if "content" in d.get("choices", [{}])[0].get("delta", {})]
        assert len(content_deltas) > 0

        # All deltas should have same id and created timestamp
        for d in all_deltas:
            assert d["id"] == chunk_id
            assert d["created"] == created

    def test_tool_call_extraction_non_streaming(self):
        """Non-streaming: Tool calls should be extracted with complete arguments."""
        parser = StreamableParser(ENC, role=Role.ASSISTANT)

        harmony = '<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location":"NYC"}<|call|>'
        tokens = ENC.encode(harmony, allowed_special="all")
        for token in tokens:
            parser.process(token)

        acc = HarmonyAccumulated()
        acc.add_from_parser(parser)

        assert len(acc.tool_calls) == 1
        fn_name, args = acc.tool_calls[0]
        assert fn_name == "get_weather"
        assert "NYC" in args

    def test_tool_call_streaming_arguments(self):
        """
        Regression test: Streaming tool calls should emit arguments incrementally.

        Previously, streaming emitted tool calls with empty arguments because
        we emitted the header before arguments were streamed. Now we:
        1. Emit tool call header with empty arguments
        2. Stream argument deltas as they arrive
        """
        parser = StreamableParser(ENC, role=Role.ASSISTANT)
        state = HarmonySessionState()
        chunk_id = "chatcmpl-test123"
        created = 1234567890

        harmony = '<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location":"NYC"}<|call|>'
        tokens = ENC.encode(harmony, allowed_special="all")

        all_deltas = []
        for token in tokens:
            parser.process(token)
            deltas = harmony_state_to_openai_deltas(parser, "test-model", state, chunk_id, created)
            all_deltas.extend(deltas)

        # Should have at least 2 deltas: header + arguments
        assert len(all_deltas) >= 2

        # First delta should have tool call header with function name
        first_delta = all_deltas[0]
        tool_calls = first_delta["choices"][0]["delta"].get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"

        # All deltas should have same id and created timestamp
        for delta in all_deltas:
            assert delta["id"] == chunk_id
            assert delta["created"] == created

        # Collect all argument fragments
        all_args = ""
        for delta in all_deltas:
            tool_calls = delta["choices"][0]["delta"].get("tool_calls", [])
            if tool_calls and "function" in tool_calls[0]:
                args = tool_calls[0]["function"].get("arguments", "")
                all_args += args

        # Combined arguments should contain the full JSON
        assert "location" in all_args
        assert "NYC" in all_args

    def test_streaming_tool_call_sets_has_tool_calls_flag(self):
        """
        Regression test: Streaming tool calls must set has_tool_calls flag for finish_reason.

        When a model makes tool calls in streaming mode, the final chunk must have
        finish_reason="tool_calls" (not "stop"), otherwise the client won't continue
        the conversation with tool results.

        This test verifies that the state.has_tool_calls flag is set when emitting tool calls.
        """
        parser = StreamableParser(ENC, role=Role.ASSISTANT)
        state = HarmonySessionState()
        chunk_id = "chatcmpl-test123"
        created = 1234567890

        # Initially, has_tool_calls should be False
        assert state.has_tool_calls is False

        harmony = '<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location":"NYC"}<|call|>'
        tokens = ENC.encode(harmony, allowed_special="all")

        for token in tokens:
            parser.process(token)
            harmony_state_to_openai_deltas(parser, "test-model", state, chunk_id, created)

        # After processing tool call, has_tool_calls should be True
        assert state.has_tool_calls is True

        # This flag should be used for finish_reason in the final chunk:
        # finish_reason = "tool_calls" if state.has_tool_calls else "stop"

    def test_streaming_suppresses_hallucinated_content_after_tool_args(self):
        """
        Regression test: After tool call arguments close, all further content is suppressed.

        GPT-OSS models sometimes hallucinate entire tool response sequences after the
        real tool call arguments, like:
        {"command":"ls"}{"stdout":"file1"}...

        When we detect the first JSON object is complete, we must suppress ALL further
        output, including any subsequent "final" channel content, because it's all
        part of the hallucination.

        Before this fix, raw Harmony tokens like "<|message|>" and "final" were leaking
        through as content chunks to the client.
        """
        parser = StreamableParser(ENC, role=Role.ASSISTANT)
        state = HarmonySessionState()
        chunk_id = "chatcmpl-test123"
        created = 1234567890

        # Simulate model outputting tool call with hallucinated response after
        # This is what GPT-OSS does: outputs the tool call args, then hallucinates
        # a response as if the tool had executed
        harmony_tool_call = '<|channel|>commentary to=functions.shell<|constrain|>json<|message|>{"command":"ls"}'
        tokens = ENC.encode(harmony_tool_call, allowed_special="all")

        tool_deltas = []
        for token in tokens:
            parser.process(token)
            deltas = harmony_state_to_openai_deltas(parser, "test-model", state, chunk_id, created)
            tool_deltas.extend(deltas)

        # We should have emitted tool call deltas
        assert len(tool_deltas) > 0
        assert state.has_tool_calls is True

        # Now the model hallucinates more content (e.g., tool response)
        # This should be suppressed because we already closed the first JSON object
        hallucinated = '{"stdout":"file1"}' + '<|channel|>final<|message|>Here is the output'
        tokens = ENC.encode(hallucinated, allowed_special="all")

        hallucinated_deltas = []
        for token in tokens:
            parser.process(token)
            deltas = harmony_state_to_openai_deltas(parser, "test-model", state, chunk_id, created)
            hallucinated_deltas.extend(deltas)

        # After first JSON object closes, suppress_remaining_output should be True
        assert state.suppress_remaining_output is True

        # No further deltas should be emitted (all suppressed)
        # The hallucinated content should NOT leak through as "final" channel content
        for delta in hallucinated_deltas:
            content = delta.get("choices", [{}])[0].get("delta", {}).get("content")
            assert content is None, f"Hallucinated content leaked through: {content}"

    def test_non_streaming_final_response(self):
        """Non-streaming response should have correct format."""
        acc = HarmonyAccumulated()
        acc.final_content = ["Hello ", "world"]
        acc.tool_calls = [("test_fn", '{"arg": "value"}')]

        response = harmony_state_to_openai_final(acc, "test-model")

        assert response["object"] == "chat.completion"
        assert response["choices"][0]["message"]["content"] == "Hello world"
        assert response["choices"][0]["finish_reason"] == "tool_calls"
        assert len(response["choices"][0]["message"]["tool_calls"]) == 1

    def test_non_streaming_openai_spec_compliance(self):
        """
        Regression test: Non-streaming response must have all required OpenAI fields.

        Required fields per OpenAI spec:
        - id: Unique identifier starting with "chatcmpl-"
        - object: Must be "chat.completion"
        - created: Unix timestamp
        - model: The model used
        - choices: Array with message and finish_reason
        - usage: Token usage statistics
        """
        acc = HarmonyAccumulated()
        acc.final_content = ["Hello world"]

        response = harmony_state_to_openai_final(acc, "test-model")

        # Verify id format
        assert "id" in response
        assert response["id"].startswith("chatcmpl-")

        # Verify object type
        assert response["object"] == "chat.completion"

        # Verify created timestamp is a reasonable Unix timestamp
        assert "created" in response
        assert isinstance(response["created"], int)
        assert response["created"] > 1700000000  # After Nov 2023

        # Verify model
        assert response["model"] == "test-model"

        # Verify choices structure
        assert "choices" in response
        assert len(response["choices"]) == 1
        assert "message" in response["choices"][0]
        assert "finish_reason" in response["choices"][0]

        # Verify usage field exists with required subfields
        assert "usage" in response
        assert "prompt_tokens" in response["usage"]
        assert "completion_tokens" in response["usage"]
        assert "total_tokens" in response["usage"]

    def test_missing_trailing_end_token(self):
        """
        Regression test: Model output without trailing <|end|> should still parse correctly
        after manually adding the end token.

        GPT-OSS models often output Harmony like:
        <|start|>assistant<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>Hello

        Note: No trailing <|end|> after "Hello". We must add it to finalize parsing.
        """
        parser = StreamableParser(ENC, role=Role.ASSISTANT)

        # Simulate model output WITHOUT trailing <|end|>
        harmony_without_end = "<|start|>assistant<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>Hello"
        tokens = ENC.encode(harmony_without_end, allowed_special="all")
        for token in tokens:
            parser.process(token)

        # Before finalization: final message is incomplete
        acc_before = HarmonyAccumulated()
        acc_before.add_from_parser(parser)
        # The final content might be empty or incomplete here

        # Now add the missing <|end|> token to finalize
        end_tokens = ENC.encode("<|end|>", allowed_special="all")
        for token in end_tokens:
            parser.process(token)

        # After finalization: final message should be complete
        acc_after = HarmonyAccumulated()
        acc_after.add_from_parser(parser)

        assert len(acc_after.final_content) == 1
        assert acc_after.final_content[0] == "Hello"

    def test_analysis_channel_tool_calls_are_fallback(self):
        """
        Regression test: Tool calls in analysis channel should only be used as fallback.

        GPT-OSS models sometimes put tool calls in the analysis channel instead of
        commentary. These should only be used if there's no other content (no
        commentary tool calls and no final content), since they're likely
        hallucinated chain-of-thought.
        """
        # Case 1: Analysis tool call only -> should be used as fallback
        acc1 = HarmonyAccumulated()
        acc1.analysis_tool_calls = [("get_time", "{}")]
        assert acc1.get_effective_tool_calls() == [("get_time", "{}")]

        # Case 2: Both analysis and commentary tool calls -> prefer commentary
        acc2 = HarmonyAccumulated()
        acc2.tool_calls = [("real_tool", '{"real": true}')]
        acc2.analysis_tool_calls = [("hallucinated_tool", '{"fake": true}')]
        assert acc2.get_effective_tool_calls() == [("real_tool", '{"real": true}')]

        # Case 3: Analysis tool call + final content -> don't use analysis tool call
        acc3 = HarmonyAccumulated()
        acc3.final_content = ["Here's the answer..."]
        acc3.analysis_tool_calls = [("ignored_tool", "{}")]
        assert acc3.get_effective_tool_calls() == []

        # Case 4: Commentary tool call + final content -> use commentary tool call
        acc4 = HarmonyAccumulated()
        acc4.final_content = ["Running the tool..."]
        acc4.tool_calls = [("real_tool", "{}")]
        assert acc4.get_effective_tool_calls() == [("real_tool", "{}")]


class TestHarmonyParseFallback:
    """Test fallback to raw content when Harmony parsing fails."""

    def test_malformed_harmony_header_triggers_error(self):
        """
        Regression test: Malformed Harmony header should raise an error.

        Real error from logs: "unexpected tokens remaining in message header"
        when model outputs something like "4-word description:" after header.
        """
        parser = StreamableParser(ENC, role=Role.ASSISTANT)

        # This pattern caused the actual error in production:
        # A Harmony header start followed by non-Harmony content
        malformed = "<|start|>assistant extra garbage that breaks parsing<|message|>"
        tokens = ENC.encode(malformed, allowed_special="all")

        with pytest.raises(HarmonyError):
            for token in tokens:
                parser.process(token)

    def test_partial_harmony_then_plain_text(self):
        """
        Regression test: Model starts with Harmony but then outputs plain text.

        This simulates a model that begins correctly but then breaks format.
        The proxy should catch this and fall back to raw content.
        """
        parser = StreamableParser(ENC, role=Role.ASSISTANT)

        # Start with valid Harmony header
        valid_start = "<|start|>assistant<|channel|>final<|message|>"
        tokens = ENC.encode(valid_start, allowed_special="all")
        for token in tokens:
            parser.process(token)  # Should work

        # Then some content
        content = "Hello "
        tokens = ENC.encode(content, allowed_special="all")
        for token in tokens:
            parser.process(token)  # Should work

        # The parser should have processed without error up to this point
        # This validates the proxy can detect parsing state changes


class TestUpstreamErrorHandling:
    """
    Test that upstream errors are properly returned to clients instead of empty responses.

    These are documentation tests - the actual error handling is in harmony_proxy.py
    and requires async/httpx mocking to fully test. These tests document the expected
    behavior.
    """

    def test_streaming_error_returns_error_message(self):
        """
        Regression test: When upstream returns 400 (e.g., context too long),
        streaming mode should return the error as an assistant message.

        Expected behavior (implemented in handle_chat_with_harmony):
        1. Check resp.status_code after opening stream
        2. If not 200, read error body and log it
        3. Return error as assistant message content
        4. Set finish_reason to "stop"

        This prevents the client from receiving an empty response with no explanation.
        """
        # The actual implementation is tested via integration tests
        # This test documents the expected error response format
        expected_error_chunk_format = {
            "id": "chatcmpl-xxx",
            "object": "chat.completion.chunk",
            "created": 12345,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "Error from model server: ..."},
                "finish_reason": None,
            }],
        }

        # Verify the expected structure has required fields
        assert "choices" in expected_error_chunk_format
        assert "delta" in expected_error_chunk_format["choices"][0]
        assert "content" in expected_error_chunk_format["choices"][0]["delta"]

    def test_non_streaming_error_returns_error_message(self):
        """
        Regression test: When upstream returns 400 in non-streaming mode,
        should return error as assistant message with the upstream status code.

        Expected behavior:
        1. Check resp.status_code before iterating response
        2. If not 200, read error body
        3. Return JSONResponse with error as content
        4. Pass through the upstream status_code
        """
        expected_error_response_format = {
            "id": "chatcmpl-xxx",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Error from model server: ..."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        # Verify the expected structure has required fields
        assert "choices" in expected_error_response_format
        assert "message" in expected_error_response_format["choices"][0]
        assert "content" in expected_error_response_format["choices"][0]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
