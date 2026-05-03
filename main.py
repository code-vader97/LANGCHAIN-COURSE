import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
load_dotenv()

def main():
    # Read information from file with error handling
    try:
        with open("dummy-info.txt", "r") as file:
            information = file.read()
    except FileNotFoundError:
        print("Error: 'dummy-info.txt' not found.")
        return

    # Check for API token
    api_token = os.environ.get("GROQ_API_KEY")
    if not api_token:
        print("Error: GROQ_API_KEY not set in environment.")
        return

    summary_template = (
        "Given the information below about a person, please:\n"
        "1. Write a short summary.\n"
        "2. List two interesting facts about them.\n\n"
        "Information:\n{information}"
    )
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm=ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_token,
    )   

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)

if __name__ == "__main__":
    main()