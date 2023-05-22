from typing import Optional
from nebuly.core.apis import init, tracker  # type: ignore  # noqa 401
from nebuly.core.schemas import DevelopmentPhase, Task  # type: ignore  # noqa 401
import os

api_key: Optional[str] = os.getenv(key="NEBULY_API_KEY")
