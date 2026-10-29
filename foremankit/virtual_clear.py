"""
Virtual clear: trim an Anthropic /v1/messages request body once its context fills up.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def session_key(data: Dict[str, Any]) -> str:
    """
    Return the session key for the given data.

    Args:
        data: The request body data.

    Returns:
        The session key, which is the user_id if present, else the sha1 of the first user message text.
    """
    if "metadata" in data and "user_id" in data["metadata"]:
        return data["metadata"]["user_id"]
    for message in data["messages"]:
        if message["role"] == "user":
            return hashlib.sha1(_text_of(message["content"]).encode()).hexdigest()
    raise ValueError("No user messages found in data")

def _text_of(content: Any) -> str:
    """Plain text of a message content: a string as is, or the text blocks of a block list joined."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))

def fingerprint(message: Dict[str, Any]) -> str:
    """
    Return the fingerprint of the given message.

    Args:
        message: The message to fingerprint.

    Returns:
        The sha1 of the json.dumps of the message, sorted by keys.
    """
    return hashlib.sha1(json.dumps(message, sort_keys=True).encode()).hexdigest()

def valid_cut_indexes(messages: List[Dict[str, Any]]) -> List[int]:
    """
    Return the indexes of messages that are valid cut points.

    Args:
        messages: The list of messages.

    Returns:
        The list of indexes of messages that are valid cut points.
    """
    valid_indexes = []
    for i, message in enumerate(messages):
        if message["role"] == "user" and not any(
            isinstance(content, dict) and content.get("type") == "tool_result"
            for content in (message["content"] if isinstance(message["content"], list) else [message["content"]])
        ):
            valid_indexes.append(i)
    return valid_indexes

def choose_cut(messages: List[Dict[str, Any]], keep_ratio: float = 0.3) -> Optional[int]:
    """
    Choose the best cut index for the given messages.

    Args:
        messages: The list of messages.
        keep_ratio: The ratio of messages to keep.

    Returns:
        The best cut index, or None if no valid cut index is found.
    """
    valid_indexes = valid_cut_indexes(messages)
    if not valid_indexes:
        return None
    target_index = int(len(messages) * (1 - keep_ratio))
    best_index = min(valid_indexes, key=lambda i: abs(i - target_index))
    if best_index < len(messages) - 1:
        return best_index
    return None

def apply_cut(data: Dict[str, Any], handoff_text: str, cut_index: int) -> Dict[str, Any]:
    """
    Apply the cut to the given data.

    Args:
        data: The request body data.
        handoff_text: The handoff text.
        cut_index: The index of the message to cut at.

    Returns:
        The request body data with the cut applied.
    """
    if cut_index < 0 or cut_index >= len(data["messages"]):
        raise ValueError("Invalid cut index")
    new_data = data.copy()
    new_data["messages"] = [
        {"role": "user", "content": f"handoff: {handoff_text}"},
        {"role": "assistant", "content": "Understood. Continuing from the handoff."},
    ] + data["messages"][cut_index:]
    return new_data

def save_marker(state_dir: Path, key: str, message: Dict[str, Any], handoff_text: str) -> None:
    """
    Save the marker to the given state directory.

    Args:
        state_dir: The state directory.
        key: The session key.
        message: The message to fingerprint.
        handoff_text: The handoff text.
    """
    marker = {
        "key": key,
        "fingerprint": fingerprint(message),
        "handoff_text": handoff_text,
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "virtual_clear.json", "w") as file_handle:
        json.dump(marker, file_handle)

def load_marker(state_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load the marker from the given state directory.

    Args:
        state_dir: The state directory.

    Returns:
        The marker, or None if the marker is missing or corrupt.
    """
    try:
        with open(state_dir / "virtual_clear.json", "r") as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None

def reapply(data: Dict[str, Any], state_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Reapply the cut to the given data if the marker is valid.

    Args:
        data: The request body data.
        state_dir: The state directory.

    Returns:
        The request body data with the cut reapplied, or None if the marker is invalid.
    """
    marker = load_marker(state_dir)
    if not marker:
        return None
    key = session_key(data)
    if marker["key"] != key:
        return None
    for i, message in enumerate(data["messages"]):
        if fingerprint(message) == marker["fingerprint"]:
            return apply_cut(data, marker["handoff_text"], i)
    return None
