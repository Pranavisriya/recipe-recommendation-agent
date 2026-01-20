import pytest
from src.recipe_agent.nodes.rank import rank_recipes
from src.recipe_agent.state import RecipeAgentState
from src.recipe_agent.config import get_llm


def test_rank_recipes():
    llm = get_llm()

    state: RecipeAgentState = {
        "ingredients": ["rice", "vegetables", "beans", "avocado"],
        "dietary_restrictions": [],
        "max_cooking_time": 20,
        "cuisine_preference": None,
        "matched_recipes": [
            {"name": "Veggie Rice Bowl", "cuisine": "Fusion", "cooking_time": 18, "score": 4},
            {"name": "Vegetable Fried Rice", "cuisine": "Asian", "cooking_time": 15, "score": 3},
            {"name": "Mediterranean Rice Salad", "cuisine": "Mediterranean", "cooking_time": 20, "score": 2},
        ]
    }

    result = rank_recipes(state, llm=llm)

    ranked = result["matched_recipes"]
    assert len(ranked) == 3
    assert ranked[0]["name"] == "Veggie Rice Bowl"
    assert ranked[1]["name"] == "Vegetable Fried Rice"
    assert ranked[2]["name"] == "Mediterranean Rice Salad"


def test_rank_recipes_empty():
    llm = get_llm()
    state: RecipeAgentState = {
        "matched_recipes": []
    }
    result = rank_recipes(state, llm=llm)
    assert result["matched_recipes"] == []


def test_rank_recipes_partial_rank():
    llm = get_llm()

    state: RecipeAgentState = {
        "matched_recipes": [
            {"name": "Veggie Rice Bowl", "cuisine": "Fusion", "cooking_time": 18, "score": 4},
            {"name": "Vegetable Fried Rice", "cuisine": "Asian", "cooking_time": 15, "score": 3},
        ]
    }

    result = rank_recipes(state, llm=llm)
    ranked = result["matched_recipes"]
    assert ranked[0]["name"] == "Veggie Rice Bowl"
    assert ranked[1]["name"] == "Vegetable Fried Rice"


def test_rank_recipes_with_dietary_pref():
    llm = get_llm()

    state: RecipeAgentState = {
        "ingredients": ["chicken"],
        "dietary_restrictions": ["gluten-free"],
        "max_cooking_time": 30,
        "cuisine_preference": None,
        "matched_recipes": [
            {"name": "Chicken Stir Fry", "cuisine": "Asian", "cooking_time": 20, "score": 5},
            {"name": "Grilled Chicken", "cuisine": "American", "cooking_time": 25, "score": 4},
        ]
    }

    result = rank_recipes(state, llm=llm)
    ranked = result["matched_recipes"]
    assert len(ranked) == 2
    # Assuming LLM ranks based on preferences