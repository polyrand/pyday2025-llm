from typing import TypedDict


class MessageDict(TypedDict):
    role: str
    parts: list[dict[str, str]]
