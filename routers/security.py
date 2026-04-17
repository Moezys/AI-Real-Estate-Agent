import re

MAX_INPUT_LENGTH = 500

# Patterns that suggest prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)",
    r"you\s+are\s+now",
    r"system\s*:",
    r"act\s+as\s+(a\s+)?different",
    r"reveal\s+(your|the)\s+(system|instructions|prompt)",
    r"forget\s+(all|your|everything)",
    r"override\s+(your|the|all)",
    r"disregard\s+(all|your|the|previous)",
    r"new\s+instructions?\s*:",
    r"<\s*/?\s*system\s*>",
    r"\|\s*system\s*\|",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def check_injection(text: str) -> bool:
    """Return True if the text looks like a prompt injection attempt."""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return True
    return False


def sanitize_input(text: str) -> str:
    """Truncate and strip control characters from user input."""
    # Limit length
    text = text[:MAX_INPUT_LENGTH]
    # Remove null bytes and other control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def validate_user_input(text: str) -> tuple[str, bool]:
    """
    Returns (sanitized_text, is_safe).
    If is_safe is False, the input was flagged as a potential injection.
    """
    clean = sanitize_input(text)
    if not clean:
        return clean, False
    is_safe = not check_injection(clean)
    return clean, is_safe
