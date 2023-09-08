from __future__ import annotations


class ToPatch:
    """This is the class to patch."""

    def to_patch_one(self, a: int, b: float, *, c: int) -> int:
        """This is the function to patch."""
        return int(a + b + c)

    def to_patch_two(self, a: int, b: float, *, c: int) -> int:
        """This is the function to patch."""
        return int(a - b - c)
