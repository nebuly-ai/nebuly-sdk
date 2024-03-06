from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


@dataclass
class FeedbackActionMetadata:
    input: Optional[str]
    output: Optional[str]
    end_user: str
    timestamp: datetime
    anonymize: bool
    end_user_group_profile: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "end_user": self.end_user,
            "end_user_group_profile": self.end_user_group_profile,
            "timestamp": self.timestamp.isoformat(),
            "anonymize": self.anonymize,
        }


class FeedbackActionName(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    COPY_INPUT = "copy_input"
    COPY_OUTPUT = "copy_output"
    PASTE = "paste"


@dataclass
class FeedbackAction:
    slug: FeedbackActionName
    extras: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            **(self.extras or {}),
        }
