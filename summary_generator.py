# summary_generator.py
import openai
from utils import get_api_key, hash_pair, count_tokens

# Set API key for OpenAI
openai.api_key = get_api_key()

# Token limit for GPT-3.5 Turbo input
MAX_TOKENS_INPUT = 3500  # Safe buffer from actual limit

def generate_summary_gpt35(job, resume, cache_dict):
    """
    Generates or retrieves a cached GPT-3.5 summary for a given job/resume pair.
    Penalizes resumes that are unreadable or truncated.
    
    Args:
        job (str): Job description text
        resume (str): Resume text
        cache_dict (dict): Session-level summary cache
    
    Returns:
        (summary: str, penalty_factor: float)
    """
    # Case 1: Resume unreadable
    if resume is None:
        return "Unable to read resume. This candidate could not be evaluated.", 0.0

    # Truncate if too long
    penalty_factor = 1.0
    tokens = count_tokens(resume)
    if tokens > MAX_TOKENS_INPUT:
        resume = " ".join(resume.split()[:MAX_TOKENS_INPUT])
        penalty_factor = 0.8  # 20% penalty
        truncated_note = "Resume exceeded input limit. Analysis performed on truncated content.\n\n"
    else:
        truncated_note = ""

    # Cache check
    key = hash_pair(job, resume)
    if key in cache_dict:
        return cache_dict[key], penalty_factor

    # GPT prompt
    prompt = f"""
You are an AI assistant helping a recruiter. Below is a job description and a candidate's resume.
Explain in 2–3 sentences why this candidate is a good fit for the role.

Job Description:
{job}

Candidate Resume:
{resume}

Summary:
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120
        )
        summary_text = response.choices[0].message.content.strip()
        summary_text = truncated_note + summary_text

        cache_dict[key] = summary_text  # Save to cache
        return summary_text, penalty_factor
    except Exception as e:
        error_msg = f"Error generating summary: {e}"
        return error_msg, 0.0
