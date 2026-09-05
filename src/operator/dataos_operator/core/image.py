"""Shared helpers for image-text datasets and image operators."""

from __future__ import annotations

import hashlib
import io
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

IMAGE_PLACEHOLDER = "<image>"
REMOTE_IMAGE_SCHEMES = frozenset({"http", "https", "s3", "gs", "oss"})


def normalize_image_list(value: Any, *, strict: bool = True) -> list[Any]:
    """Normalize scalar/list path, PIL, or ``{bytes,path}`` images to a list."""

    if value is None:
        return []
    if _is_image_value(value):
        candidates: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        candidates = value
    else:
        if strict:
            raise ValueError("images must be an image value or a sequence of image values")
        return []

    images: list[Any] = []
    for candidate in candidates:
        if isinstance(candidate, str | Path):
            text = str(candidate).strip()
            if text:
                images.append(text)
            elif strict:
                raise ValueError("image references must not be empty")
        elif _is_embedded_image(candidate):
            images.append(candidate)
        elif strict:
            raise ValueError(f"unsupported image value: {type(candidate)!r}")
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
    reference: Any,
    *,
    media_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Decode a path, PIL image, or ``{bytes,path}`` value into stable metadata."""

    if not isinstance(reference, str | Path):
        return _inspect_embedded_image(reference)

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
    reference: Any,
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


def compute_image_hash(
    reference: Any,
    *,
    algorithm: str = "sha256",
    media_dir: str | Path | None = None,
) -> str:
    """Hash encoded bytes when present, otherwise a local file or decoded PIL pixels."""

    if isinstance(reference, str | Path):
        return compute_file_hash(reference, algorithm=algorithm, media_dir=media_dir)
    if isinstance(reference, Mapping):
        payload = reference.get("bytes")
        if isinstance(payload, bytes | bytearray | memoryview):
            digest = hashlib.new(algorithm)
            digest.update(bytes(payload))
            return digest.hexdigest()
        path = reference.get("path")
        if isinstance(path, str | Path) and str(path).strip():
            return compute_file_hash(path, algorithm=algorithm, media_dir=media_dir)
    if _is_pil_image(reference):
        digest = hashlib.new(algorithm)
        digest.update(str(reference.mode).encode())
        digest.update(f"{reference.size[0]}x{reference.size[1]}".encode())
        digest.update(reference.tobytes())
        return digest.hexdigest()
    raise ValueError(f"unsupported image value: {type(reference)!r}")


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


def _is_pil_image(value: Any) -> bool:
    return all(hasattr(value, attr) for attr in ("getbands", "mode", "size", "tobytes"))


def _is_embedded_image(value: Any) -> bool:
    if _is_pil_image(value):
        return True
    if not isinstance(value, Mapping):
        return False
    payload = value.get("bytes")
    path = value.get("path")
    return isinstance(payload, bytes | bytearray | memoryview) or isinstance(path, str | Path)


def _is_image_value(value: Any) -> bool:
    return isinstance(value, str | Path) or _is_embedded_image(value)


def _inspect_embedded_image(reference: Any) -> dict[str, Any]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required for image inspection. "
            "Install dataos-operator with the `image` extra."
        ) from exc

    encoded_bytes: int | None = None
    path: str | None = None
    close_image = False
    image = reference
    if isinstance(reference, Mapping):
        payload = reference.get("bytes")
        raw_path = reference.get("path")
        if isinstance(payload, bytes | bytearray | memoryview):
            raw = bytes(payload)
            encoded_bytes = len(raw)
            image = Image.open(io.BytesIO(raw))
            close_image = True
        elif isinstance(raw_path, str | Path) and str(raw_path).strip():
            return inspect_image(raw_path)
        else:
            raise ValueError("image mapping must contain bytes or path")
    if not _is_pil_image(image):
        raise ValueError(f"unsupported embedded image value: {type(reference)!r}")
    try:
        image.load()
        raw_width, raw_height = image.size
        orientation = image.getexif().get(274) if hasattr(image, "getexif") else None
        width, height = raw_width, raw_height
        if orientation in {5, 6, 7, 8}:
            width, height = raw_height, raw_width
        image_format = (getattr(image, "format", None) or "unknown").upper()
        result: dict[str, Any] = {
            "path": path,
            "exists": None,
            "readable": True,
            "format": image_format,
            "mime_type": Image.MIME.get(getattr(image, "format", None) or ""),
            "mode": image.mode,
            "width": width,
            "height": height,
            "pixels": width * height,
            "aspect_ratio": width / height if height else None,
            "bytes": encoded_bytes,
            "channels": len(image.getbands()),
            "has_alpha": "A" in image.getbands(),
            "is_animated": bool(getattr(image, "is_animated", False)),
            "exif_orientation": orientation,
        }
        return result
    finally:
        if close_image:
            image.close()
