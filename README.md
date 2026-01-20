# Recipe Recommendation Agent

An end-to-end **AI-powered recipe recommendation system** built using **LangChain** and **LangGraph**, following a clean, modular, agentic design.  
The system extracts ingredients and cooking time constraints from user input, searches a recipe database, ranks results intelligently, and generates personalized recipe recommendations using OpenAI models.

## Prerequisites

- Python 3.12 or higher
- uv
- API keys for:
  - OpenAI


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pranavisriya/recipe-recommendation-agent.git
cd recipe-recommendation-agent
```

2. Install `uv` in the environment if it is not present
```bash
pip install uv
```

3. Create a virtual python environment in this repo
```bash
uv init
uv venv -p 3.12
```

Any other method can also be used to create python environment.

4. Activate python environment
```bash
source .venv/bin/activate
```


5. Install dependencies using uv:
```bash
uv add -r requirements.txt
```

6. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

## Usage
Run the tests:
```bash
pytest tests/
```
Run the project:
```bash
python main.py
```


## Features

Agentic recommendation pipeline:
- Natural language ingredient and time extraction
- Recipe search over a structured recipe database
- Intelligent ranking based on:
      - Ingredient overlap
      - Cooking time feasibility
      - Cuisine diversity
- Personalized LLM-generated recommendations
- Multi-node agent workflow using LangGraph

  
## License

This project is licensed under the terms included in the LICENSE file.

## Author

Pranavi Sriya (pranavisriyavajha9@gmail.com)






