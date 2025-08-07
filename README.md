# CandidateMatch.AI

CandidateMatch.AI is a Streamlit web application that ranks resumes against a job description using OpenAI embeddings and GPT-3.5-powered fit summaries.

## Features
- Paste or load a sample job description
- Drag-and-drop up to 30 resumes (PDF)
- Auto-analyze if 30 resumes are uploaded, or manually start analysis
- Ranks candidates based on cosine similarity to the job description
- Generates AI-powered relevance summaries for each candidate
- Download results as CSV
- Reset and start over in one click

## Project Structure
```
candidate_match_ai/
├── app.py
├── resume_parser.py
├── embedding_utils.py
├── similarity.py
├── summary_generator.py
├── utils.py
├── requirements.txt
├── README.md
└── sample_data/
```
    
## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/candidate_match_ai.git
cd candidate_match_ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set OpenAI API Key

#### Local Development
Create a `.env` file:
```
OPENAI_API_KEY=sk-yourkeyhere
```

#### Streamlit Cloud
- Go to your app's **Settings → Secrets**
- Add:
```
OPENAI_API_KEY="sk-yourkeyhere"
```

### 4. Run Locally
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud
1. Push your repo to GitHub.
2. Go to Streamlit Cloud, connect your repo.
3. Add your `OPENAI_API_KEY` in **Secrets**.
4. Deploy.

## License
MIT License
