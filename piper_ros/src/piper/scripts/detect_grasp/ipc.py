#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text-framed IPC over pipes (stdin/stdout), robust to noisy log lines."""

from __future__ import annotations

import base64
import os
import pickle
import select
import time
from typing import BinaryIO, Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


_PROTO_PREFIX = b"__DGPROTO__:"
_NDARRAY_MARK = "__dg_ndarray__"
_NPSCALAR_MARK = "__dg_npscalar__"
_TUPLE_MARK = "__dg_tuple__"
_READ_BUFFERS: dict[int, bytearray] = {}
_KEEP_TAIL = len(_PROTO_PREFIX) - 1


class IPCError(RuntimeError):
    pass


def _to_bytes(line: Any) -> bytes:
    if isinstance(line, bytes):
        return line
    if isinstance(line, str):
        return line.encode("utf-8", errors="replace")
    return bytes(line)


def _encode_obj(obj: Any) -> Any:
    """Encode numpy objects into version-agnostic plain Python payload."""
    if np is not None:
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            return {
                _NDARRAY_MARK: True,
                "dtype": arr.dtype.str,
                "shape": list(arr.shape),
                "data": arr.tobytes(order="C"),
            }
        if isinstance(obj, np.generic):
            return {
                _NPSCALAR_MARK: True,
                "dtype": np.asarray(obj).dtype.str,
                "value": obj.item(),
            }

    if isinstance(obj, dict):
        return {k: _encode_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return {_TUPLE_MARK: [_encode_obj(v) for v in obj]}
    return obj


def _decode_obj(obj: Any) -> Any:
    """Decode payload produced by _encode_obj."""
    if isinstance(obj, dict):
        if obj.get(_NDARRAY_MARK) is True:
            if np is None:
                raise IPCError("numpy is required to decode ndarray payload")
            dtype = np.dtype(obj["dtype"])
            shape = tuple(obj["shape"])
            data = obj["data"]
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

        if obj.get(_NPSCALAR_MARK) is True:
            if np is None:
                return obj["value"]
            dtype = np.dtype(obj["dtype"])
            return np.asarray(obj["value"], dtype=dtype)[()]

        if _TUPLE_MARK in obj and len(obj) == 1:
            return tuple(_decode_obj(v) for v in obj[_TUPLE_MARK])

        return {k: _decode_obj(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_decode_obj(v) for v in obj]

    return obj


def send_message(stream: BinaryIO, obj: Any) -> None:
    safe_obj = _encode_obj(obj)
    payload = pickle.dumps(safe_obj, protocol=pickle.HIGHEST_PROTOCOL)
    encoded = base64.b85encode(payload)
    frame = _PROTO_PREFIX + encoded + b"\n"
    stream.write(frame)
    stream.flush()


def _decode_proto_payload(encoded: bytes) -> Any:
    payload = base64.b85decode(encoded)
    decoded = pickle.loads(payload)
    return _decode_obj(decoded)


def _decode_proto_line(raw: bytes) -> Any | None:
    line = raw.strip()
    if not line:
        return None
    idx = line.find(_PROTO_PREFIX)
    if idx < 0:
        return None
    encoded = line[idx + len(_PROTO_PREFIX):].strip()
    if not encoded:
        return None
    return _decode_proto_payload(encoded)


def _extract_from_buffer(buf: bytearray) -> Any | None:
    """Extract one protocol frame from arbitrary noisy bytes."""
    while True:
        idx = buf.find(_PROTO_PREFIX)
        if idx < 0:
            # Keep a short tail to match split prefixes across read() boundaries.
            if len(buf) > _KEEP_TAIL:
                del buf[:-_KEEP_TAIL]
            return None

        nl = buf.find(b"\n", idx + len(_PROTO_PREFIX))
        if nl < 0:
            # We found prefix but frame is incomplete: discard leading noise only.
            if idx > 0:
                del buf[:idx]
            return None

        encoded = bytes(buf[idx + len(_PROTO_PREFIX):nl]).strip()
        del buf[: nl + 1]
        if not encoded:
            continue

        try:
            return _decode_proto_payload(encoded)
        except Exception:
            # Possible false-positive prefix in logs; keep scanning.
            continue


def _recv_message_slow(stream: BinaryIO, *, timeout_s: float | None = None) -> Any:
    """Fallback path for streams without fileno (e.g., BytesIO tests)."""
    deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
    while True:
        if deadline is not None:
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise TimeoutError("Timeout waiting for IPC message")
            ready, _, _ = select.select([stream], [], [], remain)
            if not ready:
                raise TimeoutError("Timeout waiting for IPC message")

        line = stream.readline()
        if not line:
            raise EOFError("EOF")

        raw = _to_bytes(line)
        obj = _decode_proto_line(raw)
        if obj is None:
            continue
        return obj


def recv_message(stream: BinaryIO, *, timeout_s: float | None = None) -> Any:
    if not hasattr(stream, "fileno"):
        try:
            return _recv_message_slow(stream, timeout_s=timeout_s)
        except Exception as exc:  # pragma: no cover - defensive path
            raise IPCError(f"Failed to decode IPC payload: {exc}") from exc

    try:
        fd = stream.fileno()
    except Exception:
        try:
            return _recv_message_slow(stream, timeout_s=timeout_s)
        except Exception as exc:  # pragma: no cover - defensive path
            raise IPCError(f"Failed to decode IPC payload: {exc}") from exc

    buf = _READ_BUFFERS.setdefault(fd, bytearray())
    deadline = None if timeout_s is None else (time.monotonic() + timeout_s)

    while True:
        try:
            obj = _extract_from_buffer(buf)
        except Exception as exc:  # pragma: no cover - defensive path
            raise IPCError(f"Failed to decode IPC payload: {exc}") from exc
        if obj is not None:
            return obj

        if deadline is not None:
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise TimeoutError("Timeout waiting for IPC message")
            ready, _, _ = select.select([fd], [], [], remain)
            if not ready:
                raise TimeoutError("Timeout waiting for IPC message")
        else:
            ready, _, _ = select.select([fd], [], [], None)
            if not ready:  # pragma: no cover
                continue

        chunk = os.read(fd, 65536)
        if not chunk:
            raise EOFError("EOF")
        buf.extend(chunk)
