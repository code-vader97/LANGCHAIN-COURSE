import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        print("Error: HUGGINGFACEHUB_API_TOKEN not set in environment.")
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

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=api_token,
        max_new_tokens=512,
        temperature=0,
    )

    chat_model = ChatHuggingFace(llm=llm)

    chain = summary_prompt_template | chat_model
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()