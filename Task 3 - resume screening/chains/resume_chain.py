import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from prompts.templates import screening_prompt_template
import json
import re

def parse_json_safely(text):
    """Extract JSON from model output even if it has extra text around it."""
    # If it's already a dict (ChatHuggingFace sometimes returns parsed content)
    if isinstance(text, dict):
        return text
    
    # Convert to string if needed
    if hasattr(text, 'content'):
        text = text.content
    
    # Try to find JSON block in the output
    match = re.search(r'\{.*\}', str(text), re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Return a fallback if parsing fails
    return {"fit_score": None, "explanation": str(text), "extracted_data": {}}

def get_screening_chain():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    # Base LLM endpoint — must use "conversational" for featherless-ai provider
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )

    # Wrap in ChatHuggingFace for proper message formatting
    chat_llm = ChatHuggingFace(llm=llm)

    # Use StrOutputParser first, then manually parse JSON
    # (more robust than JsonOutputParser for chat models)
    chain = screening_prompt_template | chat_llm | StrOutputParser()

    return chain

def run_screening(chain, job_description, resume, config):
    """Run the chain and safely parse the result."""
    raw_output = chain.invoke(
        {"job_description": job_description, "resume": resume},
        config=config
    )
    return parse_json_safely(raw_output)