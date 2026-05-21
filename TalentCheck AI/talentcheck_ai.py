# TalentCheck AI
# Evaluator–Optimizer Resume Screening System

# ==========================================
# INSTALL DEPENDENCIES
# ==========================================
# Uncomment if running locally for the first time
# pip install openai pypdf python-dotenv

import json
import os
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "❌ OPENROUTER_API_KEY not found.\n"
        "Create a .env file and add:\n"
        "OPENROUTER_API_KEY=your_api_key"
    )

# ==========================================
# OPENROUTER CLIENT
# ==========================================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ==========================================
# PDF READER
# ==========================================
def read_pdf(file_path):
    """
    Extract text from a PDF resume.
    """

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return text.strip()


# ==========================================
# OPTIMIZER AGENT
# ==========================================
def optimizer_agent(resume_text, jd_text, feedback=None):
    """
    Generates a structured candidate evaluation.
    """

    prompt = f"""
You are the Optimizer Agent for TalentCheck AI.

Return ONLY valid JSON.

Required format:

{{
  "match_score": 0-100,
  "key_skills": ["skill1", "skill2"],
  "summary": "short explanation"
}}

RULES:
- Base all reasoning strictly on the resume
- Do NOT hallucinate skills
- Do NOT invent experience
- Keep the summary concise

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}
"""

    if feedback:
        prompt += f"""

PREVIOUS AUDIT FEEDBACK:
{feedback}

Correct the issues and regenerate the scorecard.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# ==========================================
# EVALUATOR AGENT
# ==========================================
def evaluator_agent(scorecard, resume_text):
    """
    Audits the generated scorecard for hallucinations
    and unsupported claims.
    """

    prompt = f"""
You are a strict technical auditor.

TASK:
Check for:
1. Timeline inconsistencies
2. Hallucinated skills
3. Seniority mismatch
4. Unsupported claims

RULES:
- Be strict but factual
- Do NOT over-explain
- Do NOT make assumptions
- Ignore phone numbers or irrelevant text
- Compare ONLY the resume against the scorecard

OUTPUT FORMAT:

If no issues exist:
PASS

If issues exist:
FAIL
- issue 1
- issue 2
- issue 3

RESUME:
{resume_text}

SCORECARD:
{scorecard}
"""

    response = client.chat.completions.create(
        model="google/gemma-3-4b-it:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ==========================================
# SELF-CORRECTION LOOP
# ==========================================
def refine_scorecard(resume_text, jd_text, max_iterations=3):
    """
    Runs the Optimizer → Evaluator loop
    until validation passes or retries end.
    """

    scorecard = optimizer_agent(resume_text, jd_text)

    for iteration in range(max_iterations):

        audit_result = evaluator_agent(scorecard, resume_text)

        print(f"\n===== ITERATION {iteration + 1} =====")
        print(audit_result)

        if audit_result.startswith("PASS"):

            return {
                "verified_scorecard": json.loads(scorecard),
                "status": "PASSED",
                "audit": audit_result
            }

        # Retry with evaluator feedback
        scorecard = optimizer_agent(
            resume_text=resume_text,
            jd_text=jd_text,
            feedback=audit_result
        )

    return {
        "verified_scorecard": scorecard,
        "status": "FAILED_AFTER_RETRIES",
        "audit": audit_result
    }


# ==========================================
# MAIN PROGRAM
# ==========================================
def main():

    print("\n===== TalentCheck AI =====\n")

    # Resume path
    resume_path = input("Enter resume PDF path: ").strip()

    if not os.path.exists(resume_path):
        print("❌ File not found.")
        return

    # Read resume
    resume_text = read_pdf(resume_path)

    # Job description
    print("\nPaste Job Description:\n")
    jd_text = input("JD: ")

    # Run pipeline
    result = refine_scorecard(
        resume_text=resume_text,
        jd_text=jd_text
    )

    # Display results
    print("\n===== FINAL RESULT =====\n")

    print(json.dumps(result, indent=2))


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()
