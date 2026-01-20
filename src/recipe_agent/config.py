from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .schemas import UserInput


def get_llm():
    """
    Central place to create the LLM.
    Keeps the rest of your code clean/testable.
    """
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_extractor(llm):
    """
    Structured output wrapper.
    """
    return llm.with_structured_output(UserInput)
