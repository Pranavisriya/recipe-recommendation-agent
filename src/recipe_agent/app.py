from __future__ import annotations

from langchain_core.messages import HumanMessage

from .graph import build_recipe_graph

import logging
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename="logs/app.log",   
    filemode="a"
)

logger = logging.getLogger(__name__)


def run_once(user_text: str) -> str:
    logger.info(f"OVERALL INPUT: {user_text}")

    graph = build_recipe_graph()
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=user_text)]}
    )

    messages = final_state.get("messages", [])
    final_output = messages[-1].content if messages else None

    logger.info(f"OVERALL OUTPUT: {final_output}")

    return final_output


if __name__ == "__main__":
    text = input("Tell me your ingredients/preferences: ").strip()
    print(run_once(text))
