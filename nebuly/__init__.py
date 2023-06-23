import os
from typing import Optional

from nebuly.core.apis import init, tracker  # noqa 401
from nebuly.core.schemas import DevelopmentPhase, Task  # noqa 401

api_key: Optional[str] = os.getenv(key="NEBULY_API_KEY")
