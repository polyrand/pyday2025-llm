import argparse
import os
import sys
from typing import Any
from typing import TypeAlias

from openai import OpenAI

from pyday2025_llm.constants import MODEL_NAME

Conversation: TypeAlias = list[dict]


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model_name: str = MODEL_NAME,
        max_loops: int | None = None,
        tools: list[dict] | None = None,
    ):
        self.client = client
        self.model_name = model_name
        self.max_loops = max_loops
        self.conversation_history: Conversation = []
        self.tools = tools or []

    def one_turn(self, conversation: Conversation) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=conversation,  # type: ignore
            tools=self.tools,  # type: ignore
        )
        return response

    def run(self):
        current_loop = 0

        while True:
            current_loop += 1
            print(f"Search loop iteration {current_loop}/{self.max_loops}")
            response = self.one_turn(self.conversation_history)
            self.conversation_history.append(response.choices[0].message)  # type: ignore
            if self.max_loops and current_loop >= self.max_loops:
                print("Max loops reached, stopping.")
                break

        # TODO


def main() -> int:
    """Main CLI function."""

    args = parse_args()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    user_input = args.user_input
    max_loops = args.max_loops
    model_name = args.model_name

    agent = Agent(
        client=client,
        model_name=model_name,
        max_loops=max_loops,
    )

    result = agent.run()

    print(result)

    sys.exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agent search loop to find best match for a query."
    )
    parser.add_argument(
        "--user-input",
        type=str,
        required=False,
        help="The starting user input.",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=50,
        help="Maximum number of search iterations. Default: %(default)s",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="OpenAI model name to use. Default: %(default)s",
    )

    return parser.parse_args()
