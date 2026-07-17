"""
Image Processor for LLM Vision Support
Handles conversion of uploaded images to formats compatible with vision-capable LLMs
Supports multiple LLM providers: Gemini, OpenAI, Anthropic, Qwen-VL, etc.
"""

import base64
import mimetypes
from typing import List, Dict, Any, Optional, Union
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # Headless solver/test environments
    class _HeadlessStreamlit:
        session_state = {}
        @staticmethod
        def warning(*args, **kwargs):
            return None
    st = _HeadlessStreamlit()


def _decode_base64_robust(base64_data: str) -> bytes:
    """
    Decode base64 string to bytes with padding fix and URL-safe support
    
    Args:
        base64_data: Base64 encoded string (standard or URL-safe)
    
    Returns:
        Decoded bytes
    
    Raises:
        ValueError: If decoding fails
    """
    # Remove whitespace
    base64_data = base64_data.strip()
    
    # Try standard base64 first
    try:
        # Fix padding if needed
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += '=' * (4 - missing_padding)
        return base64.b64decode(base64_data)
    except Exception:
        # Try URL-safe base64
        try:
            missing_padding = len(base64_data) % 4
            if missing_padding:
                base64_data += '=' * (4 - missing_padding)
            return base64.urlsafe_b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64: {str(e)}")


def _infer_mime_type_from_filename(filename: str) -> str:
    """
    Infer MIME type from filename using mimetypes module
    
    Args:
        filename: File name or path
    
    Returns:
        MIME type string (defaults to "image/png")
    """
    if not filename:
        return "image/png"
    
    mime_type, _ = mimetypes.guess_type(filename.lower())
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    
    # Fallback for common extensions not in mimetypes
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    ext_to_mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "svg": "image/svg+xml",
        "heic": "image/heic",
        "heif": "image/heif",
    }
    return ext_to_mime.get(ext, "image/png")


def _parse_data_uri(data_uri: str) -> tuple:
    """
    Parse data URI and extract MIME type and base64 data
    
    Args:
        data_uri: Data URI string (e.g., "data:image/png;base64,<data>")
    
    Returns:
        Tuple of (mime_type, base64_data)
    
    Raises:
        ValueError: If data URI format is invalid
    """
    if not data_uri.startswith("data:"):
        raise ValueError("Data URI must start with 'data:'")
    
    if "," not in data_uri:
        raise ValueError("Data URI must contain comma separator")
    
    header, base64_data = data_uri.split(",", 1)
    
    # Validate header format: "data:<mime>;base64" or "data:<mime>"
    if not header.startswith("data:"):
        raise ValueError("Invalid data URI header")
    
    # Extract MIME type
    mime_part = header[5:]  # Remove "data:" prefix
    if ";base64" in mime_part:
        mime_type = mime_part.split(";base64")[0]
    else:
        mime_type = mime_part.split(";")[0] if ";" in mime_part else mime_part
    
    # Validate MIME type
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Invalid or non-image MIME type: {mime_type}")
    
    return mime_type, base64_data


def extract_images_from_session() -> Optional[List[Dict[str, Any]]]:
    """
    Extract uploaded images from session state (provider-agnostic format)
    
    Returns:
        List of image dictionaries with 'mime_type' and 'data' (bytes), or None
        Format: [{"mime_type": "image/png", "data": b"..."}, ...]
        Note: base64_data is only stored if needed for specific providers
    """
    if not hasattr(st, 'session_state'):
        return None
    
    image_files = st.session_state.get("_last_image_upload")
    if not image_files or not isinstance(image_files, list):
        return None
    
    processed_images = []
    for file in image_files:
        if not isinstance(file, dict):
            continue
        
        file_data = file.get("data", "")
        if not file_data:
            continue
        
        mime_type = None
        base64_data = None
        image_bytes = None
        
        # Parse data URI or raw base64
        if "," in file_data and file_data.startswith("data:"):
            try:
                mime_type, base64_data = _parse_data_uri(file_data)
            except ValueError as e:
                # Log warning but continue processing
                try:
                    st.warning(f"Invalid data URI format, skipping image: {str(e)}")
                except:
                    pass
                continue
        else:
            # Raw base64 data (no data URI header)
            base64_data = file_data
            # Infer MIME type from filename
            file_name = file.get("name", "")
            mime_type = _infer_mime_type_from_filename(file_name)
        
        # Decode base64 to bytes
        try:
            image_bytes = _decode_base64_robust(base64_data)
        except ValueError as e:
            # Log warning but continue processing
            try:
                st.warning(f"Failed to decode image, skipping: {str(e)}")
            except:
                pass
            continue
        
        # Store image
        # For Gemini, we only need bytes, but we store base64_data for providers that need it
        # to avoid re-encoding (memory trade-off: store both for efficiency)
        img_dict = {
            "mime_type": mime_type,
            "data": image_bytes,
        }
        # Store base64_data for providers that need it (OpenAI, Anthropic, etc.)
        # This avoids re-encoding but uses more memory
        if base64_data:
            img_dict["base64_data"] = base64_data
        processed_images.append(img_dict)
    
    return processed_images if processed_images else None


def _get_base64_data(img: Dict[str, Any]) -> str:
    """
    Get base64 data from image dict, encoding from bytes if needed
    
    Args:
        img: Image dict with 'data' (bytes) and optionally base64_data
    
    Returns:
        Base64 encoded string
    """
    # If we stored the original base64, use it (more efficient)
    if "base64_data" in img:
        return img["base64_data"]
    
    # Otherwise, encode from bytes
    image_bytes = img.get("data")
    if image_bytes:
        return base64.b64encode(image_bytes).decode('utf-8')
    
    return ""


def convert_to_provider_format(images: List[Dict[str, Any]], provider: str = "gemini") -> Union[List[Dict[str, Any]], List[str], Any]:
    """
    Convert images to the format required by a specific LLM provider
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
        provider: LLM provider name ('gemini', 'openai', 'anthropic', 'qwen-vl', etc.)
    
    Returns:
        Images in the format required by the specified provider
    """
    if not images:
        return []
    
    if provider.lower() == "gemini":
        return convert_to_gemini_format(images)
    elif provider.lower() == "openai":
        return convert_to_openai_format(images)
    elif provider.lower() == "anthropic":
        return convert_to_anthropic_format(images)
    elif provider.lower() in ["qwen-vl", "qwen"]:
        return convert_to_qwen_vl_format(images)
    else:
        # Default: return in universal format (mime_type + data)
        return convert_to_gemini_format(images)


def convert_to_gemini_format(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert images to Gemini API format
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
    
    Returns:
        List of dicts in Gemini format: [{"mime_type": "...", "data": b"..."}]
    """
    gemini_images = []
    for img in images:
        gemini_images.append({
            "mime_type": img.get("mime_type", "image/png"),
            "data": img.get("data")
        })
    return gemini_images


def convert_to_openai_format(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert images to OpenAI GPT-4 Vision API format
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
    
    Returns:
        List of dicts in OpenAI format: [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
    """
    openai_images = []
    for img in images:
        mime_type = img.get("mime_type", "image/png")
        base64_data = _get_base64_data(img)
        
        data_uri = f"data:{mime_type};base64,{base64_data}"
        openai_images.append({
            "type": "image_url",
            "image_url": {
                "url": data_uri
            }
        })
    return openai_images


def convert_to_anthropic_format(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert images to Anthropic Claude API format
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
    
    Returns:
        List of dicts in Anthropic format: [{"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}]
    """
    anthropic_images = []
    for img in images:
        mime_type = img.get("mime_type", "image/png")
        base64_data = _get_base64_data(img)
        
        anthropic_images.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": base64_data
            }
        })
    return anthropic_images


def convert_to_qwen_vl_format(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert images to Qwen-VL API format (for future support)
    
    Args:
        images: List of image dicts with 'mime_type' and 'data' (bytes)
    
    Returns:
        List of dicts in Qwen-VL format (format may vary based on API)
    """
    qwen_images = []
    for img in images:
        mime_type = img.get("mime_type", "image/png")
        base64_data = _get_base64_data(img)
        
        qwen_images.append({
            "type": "image",
            "image": base64_data,
            "mime_type": mime_type
        })
    return qwen_images


def plotly_figure_to_image_dict(
    fig,
    format: str = "png",
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """
    Convert a Plotly figure to an image dict for LLM vision.

    Args:
        fig: Plotly figure (go.Figure or plotly.graph_objects.Figure)
        format: Image format ('png', 'jpeg', etc.)
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor for resolution

    Returns:
        Dict with 'mime_type' and 'data' (bytes), or None on failure.
        Compatible with convert_to_provider_format().
    """
    try:
        img_bytes = fig.to_image(format=format, width=width, height=height, scale=scale)
        mime = "image/png" if format.lower() in ("png",) else f"image/{format.lower()}"
        if format.lower() == "jpeg":
            mime = "image/jpeg"
        return {"mime_type": mime, "data": img_bytes}
    except Exception:
        return None


def figure_fingerprint(content: Union[str, Dict[str, Any]]) -> str:
    """
    Create a content-based fingerprint for a Plotly figure, ignoring non-deterministic
    fields (uid, etc.) so duplicate plots can be detected even when JSON differs.

    Args:
        content: Figure JSON as string or dict (from fig.to_json() or artifact_content)

    Returns:
        String fingerprint for comparison.
    """
    try:
        import json
        if isinstance(content, str):
            data = json.loads(content)
        elif isinstance(content, dict):
            data = content
        else:
            return ""
        parts = []
        traces = data.get("data") or []
        layout = data.get("layout") or {}
        title = (layout.get("title") or {})
        if isinstance(title, dict):
            parts.append(str(title.get("text", "")))
        else:
            parts.append(str(title))
        for t in traces:
            name = t.get("name", "")
            x = t.get("x")
            y = t.get("y")
            if x is not None and y is not None:
                try:
                    import numpy as np
                    xa, ya = np.asarray(x), np.asarray(y)
                    total = min(len(xa), len(ya))
                    n = min(total, 50)
                    if n > 0 and total > 0:
                        idx = np.linspace(0, total - 1, n, dtype=int)
                        xs = np.round(xa[idx], 4).tolist()
                        ys = np.round(ya[idx], 4).tolist()
                        parts.append(f"{name}:{xs}:{ys}")
                except Exception:
                    pass
        return "|".join(parts)
    except Exception:
        return ""


def extract_figure_data_for_agent(fig, max_points: int = 20) -> str:
    """
    Extract structured data from a Plotly figure for agent context.
    Provides trace names, axis labels, value ranges for precise physics explanation.

    Args:
        fig: Plotly figure
        max_points: Max sample points per trace (to limit token size)

    Returns:
        Human-readable summary string for the agent.
    """
    lines = []
    try:
        if hasattr(fig, "layout") and fig.layout and hasattr(fig.layout, "title"):
            title = getattr(fig.layout.title, "text", None) if fig.layout.title else None
            if title:
                lines.append(f"Plot title: {title}")

        if hasattr(fig, "layout") and fig.layout:
            xaxis = getattr(fig.layout, "xaxis", None)
            yaxis = getattr(fig.layout, "yaxis", None)
            if xaxis and hasattr(xaxis, "title") and xaxis.title:
                x_label = getattr(xaxis.title, "text", None) or xaxis.title
                x_type = getattr(xaxis, "type", "linear") or "linear"
                lines.append(f"X-axis: {x_label} ({x_type})")
            if yaxis and hasattr(yaxis, "title") and yaxis.title:
                y_label = getattr(yaxis.title, "text", None) or yaxis.title
                y_type = getattr(yaxis, "type", "linear") or "linear"
                lines.append(f"Y-axis: {y_label} ({y_type})")

        if hasattr(fig, "data") and fig.data:
            lines.append("Traces:")
            for i, trace in enumerate(fig.data):
                name = getattr(trace, "name", None) or f"Trace {i+1}"
                mode = getattr(trace, "mode", "lines") or "lines"
                x = getattr(trace, "x", None)
                y = getattr(trace, "y", None)
                if x is not None and y is not None:
                    try:
                        import numpy as np
                        x_arr, y_arr = np.asarray(x), np.asarray(y)
                        x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
                        y_min, y_max = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
                        n = len(x_arr)
                        lines.append(f"  - {name}: x∈[{x_min:.4g}, {x_max:.4g}], y∈[{y_min:.4g}, {y_max:.4g}], n={n} points")
                    except Exception:
                        lines.append(f"  - {name}: (mode={mode})")
                else:
                    lines.append(f"  - {name}: (mode={mode})")

        return "\n".join(lines) if lines else "Figure data could not be extracted."
    except Exception:
        return "Figure data could not be extracted."


def has_images_in_session() -> bool:
    """
    Check if there are images in session state
    
    Returns:
        True if images are available, False otherwise
    """
    if not hasattr(st, 'session_state'):
        return False
    
    image_files = st.session_state.get("_last_image_upload")
    return bool(image_files and isinstance(image_files, list) and len(image_files) > 0)


def clear_images_from_session():
    """Clear uploaded images from session state"""
    if hasattr(st, 'session_state'):
        if "_last_image_upload" in st.session_state:
            del st.session_state["_last_image_upload"]


def normalize_image_dict(image: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return ``{mime_type, data: bytes}`` or None if unusable."""
    if not isinstance(image, dict):
        return None
    data = image.get("data")
    if data is None:
        return None
    if isinstance(data, str):
        try:
            data = _decode_base64_robust(data)
        except Exception:
            return None
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    mime = str(image.get("mime_type") or "image/png")
    if not mime.startswith("image/"):
        mime = "image/png"
    return {"mime_type": mime, "data": bytes(data)}


def provider_supports_vision(provider_or_model: Optional[str]) -> bool:
    """Return True when the chat provider/model accepts multimodal image blocks."""
    raw = (provider_or_model or "").strip().lower()
    if not raw:
        return False
    provider = raw.split(":", 1)[0]
    model = raw.split(":", 1)[1] if ":" in raw else raw
    if provider in {"gemini", "google_genai", "google", "openai", "anthropic"}:
        return True
    if any(token in model for token in ("llava", "qwen-vl", "qwen2-vl", "vision", "gpt-4o", "gemini")):
        return True
    return False


def _append_unique_image(out: List[Dict[str, Any]], image: Optional[Dict[str, Any]], *, limit: int) -> bool:
    """Append normalized image if valid. Returns True when ``out`` is full."""
    norm = normalize_image_dict(image) if isinstance(image, dict) else None
    if not norm:
        return len(out) >= limit
    # Dedup by first 64 bytes + mime (cheap; enough for turn-level history).
    sig = (norm.get("mime_type"), (norm.get("data") or b"")[:64])
    for existing in out:
        esig = (existing.get("mime_type"), (existing.get("data") or b"")[:64])
        if esig == sig:
            return len(out) >= limit
    out.append(norm)
    return len(out) >= limit


def collect_turn_images(
    session_context: Optional[Dict[str, Any]],
    *,
    limit: int = 3,
    include_figures: bool = True,
) -> List[Dict[str, Any]]:
    """
    Collect images the agents should see for the current turn.

    Prefer explicit ``turn_images`` (just uploaded/pasted). Also include the
    latest page/agent Plotly figures so agents can see and explain plots.
    """
    ctx = session_context or {}
    out: List[Dict[str, Any]] = []
    for item in ctx.get("turn_images") or []:
        if isinstance(item, dict) and "mime_type" in item and "data" in item:
            if _append_unique_image(out, item, limit=limit):
                return out
        elif isinstance(item, dict) and isinstance(item.get("figure_image"), dict):
            if _append_unique_image(out, item["figure_image"], limit=limit):
                return out

    # Explicit chat uploads win for this turn; do not mix older artifacts in.
    if out:
        return out

    if include_figures:
        last_fig = ctx.get("last_figure_image")
        if isinstance(last_fig, dict) and _append_unique_image(out, last_fig, limit=limit):
            return out

    allowed_types = {"image", "user_image"}
    if include_figures:
        allowed_types.add("figure")
    for item in reversed(list(ctx.get("artifact_history") or [])):
        if not isinstance(item, dict):
            continue
        if item.get("type") not in allowed_types:
            continue
        fig = item.get("figure_image")
        if _append_unique_image(out, fig if isinstance(fig, dict) else None, limit=limit):
            break
    return out


def figure_text_context(session_context: Optional[Dict[str, Any]], *, limit: int = 3) -> str:
    """Text fallback describing recent plots/images when the LLM has no vision."""
    ctx = session_context or {}
    parts: List[str] = []
    last_data = ctx.get("last_figure_data")
    if isinstance(last_data, str) and last_data.strip():
        parts.append("Latest plot summary:\n" + last_data.strip())

    seen = 0
    for item in reversed(list(ctx.get("artifact_history") or [])):
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"figure", "image", "user_image"}:
            continue
        caption = str(item.get("caption") or item.get("title") or item.get("type") or "artifact").strip()
        fig_data = item.get("figure_data")
        if isinstance(fig_data, str) and fig_data.strip():
            parts.append(f"Artifact ({caption}):\n{fig_data.strip()}")
        else:
            parts.append(f"Artifact available in chat: {caption}")
        seen += 1
        if seen >= limit:
            break
    return "\n\n".join(parts)


def langchain_human_content(
    text: str,
    images: Optional[List[Dict[str, Any]]] = None,
    *,
    supports_vision: bool = True,
    text_fallback: str = "",
) -> Union[str, List[Dict[str, Any]]]:
    """
    Build LangChain HumanMessage content: plain text, or multimodal blocks
    (text + OpenAI-style image_url) when images are provided and the provider
    supports vision.
    """
    text = (text or "").strip() or "Please analyze the attached image(s)."
    fallback = (text_fallback or "").strip()
    images = images or []

    if images and supports_vision:
        if fallback and fallback not in text:
            text = text + "\n\n" + fallback
        blocks: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        blocks.extend(convert_to_openai_format(images))
        return blocks

    extras: List[str] = []
    if images and not supports_vision:
        extras.append(
            f"[{len(images)} image(s) were attached, but the current LLM provider "
            "does not support vision. Switch to Gemini for full visual analysis.]"
        )
    if fallback:
        extras.append(fallback)
    if extras:
        text = text + "\n\n" + "\n\n".join(extras)
    return text


def sanitize_message_content_for_provider(content: Any, *, supports_vision: bool) -> Any:
    """Flatten multimodal message content for text-only providers."""
    if supports_vision or not isinstance(content, list):
        return content
    texts: List[str] = []
    saw_image = False
    for block in content:
        if isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                piece = str(block.get("text") or "").strip()
                if piece:
                    texts.append(piece)
            elif btype in {"image_url", "image"}:
                saw_image = True
            else:
                piece = str(block.get("text") or "").strip()
                if piece:
                    texts.append(piece)
        else:
            piece = str(block).strip()
            if piece:
                texts.append(piece)
    if saw_image:
        texts.append(
            "[Attached image omitted — this provider does not support vision. "
            "Switch to Gemini for visual analysis.]"
        )
    return "\n".join(texts) if texts else ""


def sanitize_messages_for_provider(messages: List[Any], provider_or_model: Optional[str]) -> List[Any]:
    """Strip multimodal content parts when the provider is text-only."""
    supports = provider_supports_vision(provider_or_model)
    if supports:
        return list(messages or [])
    out: List[Any] = []
    for message in messages or []:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            out.append(message)
            continue
        flat = sanitize_message_content_for_provider(content, supports_vision=False)
        if hasattr(message, "model_copy"):
            out.append(message.model_copy(update={"content": flat}))
        else:
            out.append(message)
    return out
