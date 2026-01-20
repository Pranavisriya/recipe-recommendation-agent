from __future__ import annotations

from typing import Annotated, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage


class RecipeAgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ingredients: list[str]
    dietary_restrictions: list[str]
    max_cooking_time: int  # minutes
    cuisine_preference: str
    matched_recipes: list[dict]
