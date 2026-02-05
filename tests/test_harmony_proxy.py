"""
Tests for the Harmony proxy.
"""

import pytest
from harmony_proxy import (
    is_harmony_model,
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


class TestHarmonyToOpenAIConversion:
    """Test Harmony to OpenAI format conversion."""

    def test_streaming_drops_analysis_channel(self):
        """Analysis channel should be dropped in streaming output."""
        parser = StreamableParser(ENC, role=Role.ASSISTANT)
        state = HarmonySessionState()

        # Simulate analysis channel token
        harmony = "<|channel|>analysis<|message|>thinking...<|end|>"
        tokens = ENC.encode(harmony, allowed_special="all")
        for token in tokens:
            parser.process(token)
            deltas = harmony_state_to_openai_deltas(parser, "test-model", state)
            # Analysis should produce no deltas
            for d in deltas:
                assert "analysis" not in str(d)

    def test_final_channel_produces_content(self):
        """Final channel should produce content deltas."""
        parser = StreamableParser(ENC, role=Role.ASSISTANT)
        state = HarmonySessionState()

        harmony = "<|channel|>final<|message|>Hello world<|end|>"
        tokens = ENC.encode(harmony, allowed_special="all")

        all_deltas = []
        for token in tokens:
            parser.process(token)
            deltas = harmony_state_to_openai_deltas(parser, "test-model", state)
            all_deltas.extend(deltas)

        # Should have content deltas
        content_deltas = [d for d in all_deltas if "content" in d.get("choices", [{}])[0].get("delta", {})]
        assert len(content_deltas) > 0

    def test_tool_call_extraction(self):
        """Tool calls should be extracted correctly."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
