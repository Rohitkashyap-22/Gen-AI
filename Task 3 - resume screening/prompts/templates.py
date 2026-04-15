from langchain_core.prompts import PromptTemplate

# Prompt Engineering: Clear instructions and output constraints
SCREENING_PROMPT = """
You are an expert technical recruiter evaluating a candidate for a role.
Compare the candidate's resume against the job description below.

Job Description:
{job_description}

Candidate Resume:
{resume}

Process Pipeline Steps:
1. Skill Extraction: Extract the candidate's Skills, Experience, and Tools.
2. Matching Logic: Compare the extracted data with the job requirements.
3. Scoring: Assign a fit score from 0 to 100.
4. Explanation: Provide reasoning for why this score was assigned.

CRITICAL RULE: Do NOT assume skills not present in the resume.

Output your evaluation strictly in the following JSON format:
{{
    "extracted_data": {{
        "skills": [],
        "experience": "",
        "tools": []
    }},
    "fit_score": 0,
    "explanation": ""
}}
"""

# Using PromptTemplate as required
screening_prompt_template = PromptTemplate(
    input_variables=["job_description", "resume"],
    template=SCREENING_PROMPT
)