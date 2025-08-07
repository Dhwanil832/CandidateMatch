# resume_parser.py
import fitz  # PyMuPDF
import spacy

# Load spaCy model for name detection (English)
# Make sure to run: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file) -> str:
    """
    Extracts text from an uploaded PDF file using PyMuPDF.
    Returns plain text.
    """
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text("text")
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_candidate_name(text: str) -> str:
    """
    Extracts candidate name from resume text using spaCy NER.
    Falls back to first non-empty line if no PERSON entity is found.
    """
    if not text:
        return "Unknown"

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:  # Avoid long matches
            return ent.text.strip()

    # Fallback: First non-empty line
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line
    return "Unknown"

def parse_resume(uploaded_file):
    """
    Takes an uploaded file object from Streamlit,
    returns dict with filename, text, and candidate name.
    If unreadable, text is None and name is 'Unknown'.
    """
    text = extract_text_from_pdf(uploaded_file)
    if not text:
        return {
            "filename": uploaded_file.name,
            "text": None,
            "name": "Unknown"
        }

    name = extract_candidate_name(text)
    return {
        "filename": uploaded_file.name,
        "text": text,
        "name": name
    }
