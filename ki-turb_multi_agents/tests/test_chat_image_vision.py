"""User-pasted chat images and page plots must reach agents safely per provider."""

from langchain_core.messages import HumanMessage

from agents.shared.image_processor import (
    collect_turn_images,
    figure_text_context,
    langchain_human_content,
    normalize_image_dict,
    provider_supports_vision,
    sanitize_messages_for_provider,
)


PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_normalize_and_langchain_blocks_for_vision_provider():
    img = normalize_image_dict({"mime_type": "image/png", "data": PNG})
    assert img is not None
    content = langchain_human_content("What is in this figure?", [img], supports_vision=True)
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_langchain_content_strips_images_for_deepseek():
    img = normalize_image_dict({"mime_type": "image/png", "data": PNG})
    content = langchain_human_content(
        "Explain this spectrum",
        [img],
        supports_vision=False,
        text_fallback="Latest plot summary:\nX-axis: k",
    )
    assert isinstance(content, str)
    assert "image_url" not in content
    assert "does not support vision" in content
    assert "Latest plot summary" in content


def test_provider_supports_vision_matrix():
    assert provider_supports_vision("gemini") is True
    assert provider_supports_vision("google_genai:gemini-2.5-flash") is True
    assert provider_supports_vision("deepseek") is False
    assert provider_supports_vision("deepseek:deepseek-v4-pro") is False
    assert provider_supports_vision("ollama:qwen2.5-coder:32b") is False


def test_collect_turn_images_prefers_turn_images():
    png = b"\x89PNG\r\n\x1a\nfake"
    ctx = {
        "turn_images": [{"mime_type": "image/png", "data": png}],
        "artifact_history": [
            {
                "type": "image",
                "figure_image": {"mime_type": "image/jpeg", "data": b"other"},
                "caption": "older",
            }
        ],
    }
    found = collect_turn_images(ctx)
    assert len(found) == 1
    assert found[0]["data"] == png


def test_collect_turn_images_includes_page_figures():
    ctx = {
        "last_figure_image": {"mime_type": "image/png", "data": b"page-plot"},
        "artifact_history": [
            {
                "type": "figure",
                "figure_image": {"mime_type": "image/png", "data": b"agent-plot"},
                "caption": "Energy spectrum",
            },
            {
                "type": "image",
                "figure_image": {"mime_type": "image/png", "data": b"user-upload"},
                "caption": "pasted",
            },
        ],
    }
    found = collect_turn_images(ctx, include_figures=True, limit=3)
    datas = [item["data"] for item in found]
    assert b"page-plot" in datas
    assert b"user-upload" in datas or b"agent-plot" in datas


def test_collect_turn_images_falls_back_to_artifact_history():
    ctx = {
        "artifact_history": [
            {
                "type": "figure",
                "figure_image": {"mime_type": "image/png", "data": b"plot"},
            },
            {
                "type": "image",
                "figure_image": {"mime_type": "image/png", "data": b"user-upload"},
                "caption": "pasted",
            },
        ]
    }
    found = collect_turn_images(ctx, include_figures=False)
    assert len(found) == 1
    assert found[0]["data"] == b"user-upload"


def test_figure_text_context_uses_last_figure_data():
    text = figure_text_context({
        "last_figure_data": "Plot title: Energy spectrum\nX-axis: k (log)",
        "artifact_history": [{"type": "figure", "caption": "norm", "figure_data": "y∈[1,2]"}],
    })
    assert "Energy spectrum" in text
    assert "norm" in text


def test_sanitize_messages_for_deepseek_flattens_image_blocks():
    msg = HumanMessage(content=[
        {"type": "text", "text": "see plot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ])
    cleaned = sanitize_messages_for_provider([msg], "deepseek:deepseek-v4-pro")
    assert len(cleaned) == 1
    assert isinstance(cleaned[0].content, str)
    assert "see plot" in cleaned[0].content
    assert "image_url" not in cleaned[0].content
    assert "does not support vision" in cleaned[0].content
