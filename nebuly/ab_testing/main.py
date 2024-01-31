from collections.abc import Sequence
from typing import Optional
from uuid import UUID

from nebuly.ab_testing.types import Response


class ABTesting:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    def get_variants(
        self, user: str, project_id: UUID, feature_flags: Sequence[str]
    ) -> Response:
        """
        Get the variant for each feature flag for a given user
        """
        raise NotImplementedError
