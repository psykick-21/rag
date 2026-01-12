from typing import Any
from datetime import datetime
import uuid

def make_json_serializable(obj: Any) -> Any:
    """Converts UUID and datetime objects to JSON-serializable types."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj