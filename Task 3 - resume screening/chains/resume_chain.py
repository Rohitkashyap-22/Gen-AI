from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from prompts.templates import screening_prompt_template

def get_screening_chain():
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Use a JSON parser for structured output (Bonus Feature)
    json_parser = JsonOutputParser()

    # LCEL (LangChain Expression Language) Pipeline
    # Architecture: Prompt -> LLM -> Parse
    chain = screening_prompt_template | llm | json_parser

    return chain