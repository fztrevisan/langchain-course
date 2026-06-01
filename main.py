import os

from dotenv import load_dotenv

load_dotenv()


def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    print(f"Olha a chave {openai_api_key}")


if __name__ == "__main__":
    main()
