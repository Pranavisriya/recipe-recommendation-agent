import pytest
from src.recipe_agent.nodes.extract import extract_user_preferences
from src.recipe_agent.state import RecipeAgentState
from src.recipe_agent.config import get_llm, get_extractor


def test_extract_user_preferences():
    llm = get_llm()
    extractor = get_extractor(llm)

    state: RecipeAgentState = {
        "messages": [{"role": "user", "content": "I have rice, vegetables, beans, avocado and 20 minutes."}]
    }

    result = extract_user_preferences(state, extractor=extractor)

    assert result["ingredients"] == ["rice", "vegetables", "beans", "avocado"]
    assert result["dietary_restrictions"] == []
    assert result["max_cooking_time"] == 20
    assert result["cuisine_preference"] is None


def test_extract_user_preferences_with_dietary():
    llm = get_llm()
    extractor = get_extractor(llm)

    state: RecipeAgentState = {
        "messages": [{"role": "user", "content": "I have chicken, rice, and I'm vegan with 30 minutes."}]
    }

    result = extract_user_preferences(state, extractor=extractor)

    assert "chicken" in result["ingredients"]
    assert "rice" in result["ingredients"]
    assert "vegan" in result["dietary_restrictions"]
    assert result["max_cooking_time"] == 30


def test_extract_user_preferences_with_cuisine():
    llm = get_llm()
    extractor = get_extractor(llm)

    state: RecipeAgentState = {
        "messages": [{"role": "user", "content": "I have pasta, tomatoes, basil, and I want Italian food in 25 minutes."}]
    }

    result = extract_user_preferences(state, extractor=extractor)

    assert "pasta" in result["ingredients"]
    assert "Italian" == result["cuisine_preference"]
    assert result["max_cooking_time"] == 25