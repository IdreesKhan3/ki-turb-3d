from agents.langgraph.structured_output import supports_agent_response_format


def test_deepseek_uses_prompt_parser_fallback():
    assert supports_agent_response_format("deepseek:deepseek-v4-pro") is False
    assert supports_agent_response_format("ollama:qwen2.5-coder:32b") is True
    assert supports_agent_response_format("google_genai:gemini-2.5-flash") is True
