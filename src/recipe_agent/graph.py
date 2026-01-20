from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import RecipeAgentState
from .config import get_llm, get_extractor

from .nodes.extract import extract_user_preferences
from .nodes.search import search_recipes
from .nodes.rank import rank_recipes
from .nodes.recommend import generate_recommendation


def build_recipe_graph():
    llm = get_llm()
    extractor = get_extractor(llm)

    g = StateGraph(RecipeAgentState)

    # Inject llm/extractor via lambdas (simple dependency injection)
    g.add_node("extract_user_preferences", lambda s: extract_user_preferences(s, extractor=extractor))
    g.add_node("search_recipes", search_recipes)
    g.add_node("rank_recipes", lambda s: rank_recipes(s, llm=llm))
    g.add_node("generate_recommendation", lambda s: generate_recommendation(s, llm=llm))

    g.set_entry_point("extract_user_preferences")
    g.add_edge("extract_user_preferences", "search_recipes")
    g.add_edge("search_recipes", "rank_recipes")
    g.add_edge("rank_recipes", "generate_recommendation")
    g.add_edge("generate_recommendation", END)

    return g.compile()
