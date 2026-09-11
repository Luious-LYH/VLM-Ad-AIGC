from __future__ import annotations

import base64
import binascii
import ipaddress
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .config import Settings


class UnsafeAssetUri(ValueError):
    pass


def _decode_repeated(value: str) -> str:
    previous = value
    for _ in range(3):
        decoded = unquote(previous)
        if decoded == previous:
            return decoded
        previous = decoded
    return previous


def _host_allowed(host: str, patterns: tuple[str, ...]) -> bool:
    normalized = host.rstrip(".").lower()
    for pattern in patterns:
        candidate = pattern.rstrip(".").lower()
        if candidate.startswith("*."):
            suffix = candidate[1:]
            if normalized.endswith(suffix) and normalized != suffix[1:]:
                return True
        elif normalized == candidate:
            return True
    return False


def _validate_data_url(uri: str, cfg: Settings) -> str:
    if not cfg.allow_data_urls:
        raise UnsafeAssetUri("data URLs are disabled")
    match = re.fullmatch(
        r"data:(image/(?:png|jpeg|webp)|video/(?:mp4|webm));base64,([A-Za-z0-9+/=\r\n]+)",
        uri,
    )
    if not match:
        raise UnsafeAssetUri("only base64 PNG/JPEG/WebP/MP4/WebM data URLs are accepted")
    payload = match.group(2)
    estimated_bytes = len(payload.rstrip("=")) * 3 // 4
    if estimated_bytes > cfg.max_data_url_bytes:
        raise UnsafeAssetUri("data URL exceeds configured size limit")
    try:
        base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise UnsafeAssetUri("invalid base64 data URL") from exc
    return uri


def validate_asset_uri(uri: str, cfg: Settings) -> str:
    value = uri.strip()
    if not value:
        raise UnsafeAssetUri("asset URI cannot be empty")
    decoded = _decode_repeated(value)
    if "\x00" in decoded:
        raise UnsafeAssetUri("asset URI contains a NUL byte")
    if decoded.lower().startswith("data:"):
        return _validate_data_url(decoded, cfg)

    # urlsplit treats a Windows drive letter as a URI scheme ("C:"). Handle
    # drive-qualified paths before applying the network-scheme policy.
    windows_drive_path = bool(re.match(r"^[A-Za-z]:[\\/]", decoded))
    parsed = urlsplit(decoded)
    if parsed.scheme.lower() in {"http", "https"}:
        if parsed.username or parsed.password:
            raise UnsafeAssetUri("credentials in asset URLs are forbidden")
        if parsed.fragment:
            raise UnsafeAssetUri("URL fragments are forbidden")
        host = parsed.hostname
        if not host or not _host_allowed(host, cfg.url_allowlist):
            raise UnsafeAssetUri(f"asset URL host is not allowlisted: {host or '<missing>'}")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address and not _host_allowed(host, cfg.url_allowlist):
            raise UnsafeAssetUri("IP-literal URL is not explicitly allowlisted")
        return decoded
    if parsed.scheme and not windows_drive_path:
        raise UnsafeAssetUri(f"unsupported asset URI scheme: {parsed.scheme}")

    candidate = Path(decoded)
    if not candidate.is_absolute():
        raise UnsafeAssetUri("local asset paths must be absolute")
    resolved = candidate.resolve(strict=False)
    if not any(resolved == root or root in resolved.parents for root in cfg.asset_roots):
        raise UnsafeAssetUri("local asset path escapes configured asset roots")
    return str(resolved)
