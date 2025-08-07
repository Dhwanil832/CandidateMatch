# similarity.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def rank_candidates(job_embedding, resumes_data):
    """
    Ranks resumes by cosine similarity to the job description.
    
    Args:
        job_embedding (list): Embedding vector for the job description.
        resumes_data (list): List of dicts with keys:
            - 'filename'
            - 'name'
            - 'text'
            - 'embedding'

    Returns:
        pd.DataFrame: Ranked candidates with columns:
            Candidate Name, File Name, Similarity Score
    """
    results = []

    for r in resumes_data:
        if r['embedding'] is None or r['text'] is None:
            score = 0.0
        else:
            score = cosine_similarity(
                [job_embedding], [r['embedding']]
            )[0][0]

        results.append({
            "Candidate Name": r['name'],
            "File Name": r['filename'],
            "Similarity Score": round(score * 100, 2)  # As percentage
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)
    return df
