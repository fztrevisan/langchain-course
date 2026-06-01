import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from input_data import INFORMATION

load_dotenv()


def main():
    summary_template_str = """## Action
Based on the info provided, create:
1. A short summary;
2. Two interesting facts about the person.

## Information
{information}
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template_str
    )

    # llm = ChatOpenAI(
    #     temperature=0,
    #     model="gpt-4o",
    #     base_url="https://models.inference.ai.azure.com",
    #     api_key=os.getenv("GITHUB_GPT_PAT"),
    # )
    # models: qwen3.5:4b-mlx | gemma4:e4b-mlx
    llm = ChatOllama(
        temperature=0,
        model="gemma4:e4b-mlx",
        reasoning=False,
    )
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": INFORMATION})

    token_usage = response.response_metadata.get("token_usage", {})
    print("Total Tokens:", token_usage.get("total_tokens"))
    print("Prompt Tokens:", token_usage.get("prompt_tokens"))
    print("Completion Tokens:", token_usage.get("completion_tokens"))

    response.pretty_print()


if __name__ == "__main__":
    main()
