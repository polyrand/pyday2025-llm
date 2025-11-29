import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Result of a tool execution."""

    status_code: int
    output: str
    error_message: str | None = None

    def __post_init__(self):
        if self.status_code != 0 and not self.error_message:
            raise ValueError(
                "error_message must be provided if status_code is non-zero"
            )


def pydantic_to_tool_params(
    name: str,
    description: str,
    parameters_model: type[BaseModel],
) -> dict:
    """Convert a Pydantic model to tool parameters dict."""

    is_pydantic = isinstance(parameters_model, type) and issubclass(
        parameters_model, BaseModel
    )
    if not is_pydantic:
        raise ValueError("parameters_model must be a Pydantic model class")

    model_schema = parameters_model.model_json_schema()
    if "title" in model_schema:
        # Remove title to avoid issues with some LLM parsers
        del model_schema["title"]

    # For strict mode, all properties must be in required array
    # and additionalProperties must be false
    # https://platform.openai.com/docs/guides/function-calling#strict-mode
    if "properties" in model_schema:
        model_schema["required"] = list(model_schema["properties"].keys())
    model_schema["additionalProperties"] = False

    return {
        "name": name,
        "description": description,
        "parameters": model_schema,
        "strict": True,
    }


# TASK 1: Implement list_files tools parameters model
class ListFilesParams(BaseModel):
    folder: str


ListFilesToolDefinition = pydantic_to_tool_params(
    name="list_files",
    description="List top-level files in a folder (non-recursive). ",
    parameters_model=ListFilesParams,
)


# def main():
#     from rich.pretty import pprint

#     pprint(ListFilesToolDefinition)


def list_files(folder: Path) -> ToolResult:
    """List top-level files in a folder (non-recursive)."""
    try:
        return ToolResult(
            output=json.dumps([p for p in folder.iterdir()], default=str),
            status_code=0,
        )
    except Exception as e:
        return ToolResult(status_code=1, output="", error_message=str(e))


# TODO (1): Implement read_file tool

# TODO (2): Improve read_file error codes

# TODO (3): Implement grep_pattern tool

# TODO (4): Add glob import to grep_pattern tool

# TODO (5): Add ask_user tool
