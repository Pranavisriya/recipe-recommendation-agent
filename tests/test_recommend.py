import pytest
from src.recipe_agent.nodes.recommend import generate_recommendation
from src.recipe_agent.state import RecipeAgentState
from src.recipe_agent.config import get_llm
from langchain_core.messages import AIMessage


def test_generate_recommendation():
    llm = get_llm()

    state: RecipeAgentState = {
        "ingredients": ["rice", "vegetables", "beans", "avocado"],
        "matched_recipes": [
            {"name": "Veggie Rice Bowl", "cooking_time": 18, "cuisine": "Fusion"},
            {"name": "Vegetable Fried Rice", "cooking_time": 15, "cuisine": "Asian"},
            {"name": "Mediterranean Rice Salad", "cooking_time": 20, "cuisine": "Mediterranean"},
        ]
    }

    result = generate_recommendation(state, llm=llm)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "Veggie Rice Bowl" in result["messages"][0].content


def test_generate_recommendation_no_ingredients():
    llm = get_llm()
    state: RecipeAgentState = {
        "ingredients": [],
        "matched_recipes": []
    }

    result = generate_recommendation(state, llm=llm)

    assert result["messages"][0].content == "Tell me what ingredients you have so I can recommend recipes."


def test_generate_recommendation_no_recipes():
    llm = get_llm()
    state: RecipeAgentState = {
        "ingredients": ["rice"],
        "matched_recipes": []
    }

    result = generate_recommendation(state, llm=llm)

    assert "I couldn’t find a matching recipe" in result["messages"][0].content


def test_generate_recommendation_few_recipes():
    llm = get_llm()
    state: RecipeAgentState = {
        "ingredients": ["pasta"],
        "matched_recipes": [
            {"name": "Pasta Primavera", "cooking_time": 15, "cuisine": "Italian"},
            {"name": "Spaghetti Bolognese", "cooking_time": 30, "cuisine": "Italian"},
        ]
    }

    result = generate_recommendation(state, llm=llm)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "Pasta Primavera" in result["messages"][0].content