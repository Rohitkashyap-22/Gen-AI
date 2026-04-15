import os
from dotenv import load_dotenv
from chains.resume_chain import get_screening_chain

# Load environment variables to enable LangSmith tracing
load_dotenv()

# 1. Inputs: 1 Job Description
job_description = """
Data Scientist role requiring 3+ years of experience.
Must have strong skills in Python, Machine Learning, SQL, and Deep Learning.
Familiarity with LangChain, NLP, and AWS is a huge plus.
"""

# 1. Inputs: At least 3 resumes
resumes = {
    "Strong": "5 years of experience as a Data Scientist. Expert in Python, SQL, Machine Learning, and Deep Learning. Built multiple NLP pipelines using LangChain and deployed them on AWS.",
    "Average": "2 years of experience as a Data Analyst. Good knowledge of Python and SQL. Built some basic Machine Learning models using scikit-learn. Currently learning LangChain.",
    "Weak": "Recent graduate with a degree in biology. Worked in a lab for 1 year. Basic knowledge of Excel and data entry. Looking to transition into tech. No coding experience."
}

def main():
    # Initialize the modular chain
    chain = get_screening_chain()

    # Run the pipeline for each candidate
    for candidate_type, resume in resumes.items():
        print(f"--- Evaluating {candidate_type} Candidate ---")

        # Bonus: Use LangSmith tags to categorize runs
        config = {"tags": [candidate_type.lower()]}

        # Use the .invoke() method as required
        response = chain.invoke(
            {"job_description": job_description, "resume": resume},
            config=config
        )

        # Output the results
        print(f"Fit Score: {response.get('fit_score')}")
        print(f"Explanation: {response.get('explanation')}")
        print(f"Extracted Skills: {response.get('extracted_data', {}).get('skills')}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()