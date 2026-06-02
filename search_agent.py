import logging

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(module)s][%(funcName)s]: %(message)s"'
)
load_dotenv()


@tool
def search(query: str) -> str:
    """Tool that searches over the internet

    Args:
        query (str): The query to search for

    Returns:
        str: Search result
    """
    logging.info(f'Searching for: {query}')
    return 'Edmonton/CA weather is rainny and cold!!!'


@tool
def weather(query: str) -> str:
    """Tool that searches the wheather based on the user location.
    If the query contains a location, that one should be used instead of the current location.

    Args:
        query (str): User query for the weather somewhere.

    Returns:
        str: Internet search response with the weather.
    """
    logging.info(f'Searching weather for query: {query}')
    return 'Edmonton/CA weather is rainny and cold'


def main():

    llm = ChatOllama(
        temperature=0,
        model="qwen3.5:4b-mlx",
        reasoning=False,
    )

    tools = [weather]

    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke(
        {'messages': HumanMessage('What is the weather here?')}
    )
    print(response)

if __name__ == '__main__':
    main()