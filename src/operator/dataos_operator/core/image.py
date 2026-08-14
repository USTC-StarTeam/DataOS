"""Shared helpers for image-text datasets and image operators."""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

IMAGE_PLACEHOLDER = "<image>"
REMOTE_IMAGE_SCHEMES = frozenset({"http", "https", "s3", "gs", "oss"})


def normalize_image_list(value: Any, *, strict: bool = True) -> list[str]:
    """Normalize one image reference or a sequence of references to strings."""

    if value is None:
        return []
    if isinstance(value, str | Path):
        candidates: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        candidates = value
    else:
        if strict:
            raise ValueError("images must be a path string or a sequence of path strings")
        return []

    images: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str | Path):
            text = str(candidate).strip()
            if text:
                images.append(text)
            elif strict:
                raise ValueError("image references must not be empty")
        elif strict:
            raise ValueError(f"image reference must be a path string, got {type(candidate)!r}")
    return images


def is_remote_image_reference(reference: str | Path) -> bool:
    """Return whether an image reference points to a remote URI."""

    return urlparse(str(reference)).scheme.lower() in REMOTE_IMAGE_SCHEMES


def normalize_image_reference(
    reference: str | Path,
    *,
    media_dir: str | Path | None = None,
    make_absolute: bool = True,
    allow_remote: bool = True,
) -> str:
    """Normalize a local image path while preserving supported remote URIs."""

    text = str(reference).strip()
    if not text:
        raise ValueError("image reference must not be empty")

    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme in REMOTE_IMAGE_SCHEMES:
        if not allow_remote:
            raise ValueError(f"remote image references are not allowed: {text}")
        return text
    if scheme == "file":
        path = Path(unquote(parsed.path))
    elif scheme and not _looks_like_windows_drive(text):
        raise ValueError(f"unsupported image URI scheme {scheme!r}: {text}")
    else:
        path = Path(text).expanduser()

    if not path.is_absolute() and media_dir is not None:
        path = Path(media_dir).expanduser() / path
    if make_absolute:
        path = path.resolve(strict=False)
    return str(path)


def resolve_local_image_path(
    reference: str | Path,
    *,
    media_dir: str | Path | None = None,
) -> Path:
    """Resolve a local image path or raise for a remote reference."""

    if is_remote_image_reference(reference):
        raise ValueError(f"remote image inspection is not supported: {reference}")
    return Path(
        normalize_image_reference(
            reference,
            media_dir=media_dir,
            make_absolute=True,
            allow_remote=False,
        )
    )


def inspect_image(
    reference: str | Path,
    *,
    media_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Decode an image and return stable local-file metadata."""

    path = resolve_local_image_path(reference, media_dir=media_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required for image inspection. "
            "Install dataos-operator with the `image` extra."
        ) from exc

    with Image.open(path) as image:
        image.load()
        raw_width, raw_height = image.size
        orientation = image.getexif().get(274)
        width, height = raw_width, raw_height
        if orientation in {5, 6, 7, 8}:
            width, height = raw_height, raw_width
        image_format = (image.format or path.suffix.lstrip(".") or "unknown").upper()
        mime_type = Image.MIME.get(image.format or "") or mimetypes.guess_type(path.name)[0]
        result: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "readable": True,
            "format": image_format,
            "mime_type": mime_type,
            "mode": image.mode,
            "width": width,
            "height": height,
            "pixels": width * height,
            "aspect_ratio": width / height if height else None,
            "bytes": path.stat().st_size,
            "channels": len(image.getbands()),
            "has_alpha": "A" in image.getbands(),
            "is_animated": bool(getattr(image, "is_animated", False)),
            "exif_orientation": orientation,
        }
        if (width, height) != (raw_width, raw_height):
            result["raw_width"] = raw_width
            result["raw_height"] = raw_height
        return result


def inspect_image_safe(
    reference: str | Path,
    *,
    media_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect an image, returning failure metadata instead of raising."""

    try:
        return inspect_image(reference, media_dir=media_dir)
    except Exception as exc:  # noqa: BLE001 - this is the explicitly non-strict boundary
        normalized_path: str
        try:
            normalized_path = normalize_image_reference(reference, media_dir=media_dir)
        except Exception:  # noqa: BLE001 - retain the original malformed value for diagnostics
            normalized_path = str(reference)
        path = Path(normalized_path)
        return {
            "path": normalized_path,
            "exists": path.is_file() if not is_remote_image_reference(normalized_path) else None,
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def compute_file_hash(
    reference: str | Path,
    *,
    algorithm: str = "sha256",
    media_dir: str | Path | None = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute a streaming cryptographic hash for a local image file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc
    path = resolve_local_image_path(reference, media_dir=media_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_image_metadata(value: Any, *, strict: bool = True) -> list[dict[str, Any]]:
    """Normalize a metadata field to a list of mutable image metadata mappings."""

    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        if strict:
            raise ValueError("image metadata must be a sequence of mappings")
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            result.append(dict(item))
        elif strict:
            raise ValueError("each image metadata item must be a mapping")
    return result


def image_strategy_matches(results: Sequence[bool], *, strategy: str, keep_empty: bool) -> bool:
    """Apply an any/all strategy to per-image filter results."""

    if strategy not in {"any", "all"}:
        raise ValueError("strategy must be either 'any' or 'all'")
    if not results:
        return keep_empty
    return any(results) if strategy == "any" else all(results)


def count_image_placeholders(messages: Any, *, placeholder: str = IMAGE_PLACEHOLDER) -> int:
    """Count image placeholders in normalized string messages."""

    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes | bytearray):
        return 0
    count = 0
    for message in messages:
        if isinstance(message, Mapping):
            content = message.get("content", message.get("value"))
            if isinstance(content, str):
                count += content.count(placeholder)
    return count


def _looks_like_windows_drive(value: str) -> bool:
    return len(value) >= 2 and value[0].isalpha() and value[1] == ":"
