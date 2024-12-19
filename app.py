import requests
from sentence_transformers import SentenceTransformer, util
import csv

# Hugging Face API Setup
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_TOKEN = "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"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Sentence Transformer für Ähnlichkeitsberechnung
model = SentenceTransformer('all-MiniLM-L6-v2')

# Funktionen
def query_llama_short(prompt):
    """
    Funktion, um prägnante Antworten von der KI zu erhalten, ohne den Prompt oder irrelevante Inhalte.
    """
    try:
        response = requests.post(
            HF_API_URL, headers=headers, json={"inputs": prompt}
        )
        
        if response.status_code == 200:
            api_response = response.json()
            if api_response and isinstance(api_response, list) and "generated_text" in api_response[0]:
                generated_text = api_response[0]["generated_text"].strip()
                
                # Entferne den Prompt am Anfang, falls vorhanden
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                # Kürze die Antwort auf 2-3 Sätze
                sentences = generated_text.split(". ")
                short_answer = ". ".join(sentences[:3]).strip()
                
                return short_answer
            else:
                print("Error: Invalid API response format.")
                return "No Answer"
        else:
            print(f"Error {response.status_code}: {response.text}")
            return "API Error"
    except Exception as e:
        print(f"Exception during API call: {e}")
        return "Exception in API"

def clean_ki_answer(answer, prompt):
    """
    Bereinigt die KI-Antwort:
    - Entfernt Wiederholungen des Prompts.
    - Schneidet irrelevante Abschnitte wie 'read more' oder 'Citation...' ab.
    - Beschränkt die Antwort auf 2-3 Sätze.
    """
    # Entferne Wiederholungen des Prompts
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    
    # Entferne bekannte irrelevante Muster
    irrelevant_phrases = [
        "expand_more", "read more", "View Answer", 
        "(Citation", "See More", "More", "View More", 
        "Answer the following question clearly and concisely"
    ]
    for phrase in irrelevant_phrases:
        answer = answer.split(phrase)[0].strip()

    # Kürze die Antwort auf 2-3 Sätze
    sentences = answer.split(". ")
    short_answer = ". ".join(sentences[:3]).strip()
    return short_answer

def calculate_semantic_similarity(text1, text2):
    """
    Berechnet die semantische Ähnlichkeit zwischen zwei Texten.
    """
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(embeddings1, embeddings2).item()

def load_reference_answers(file_path):
    """
    Lädt die Fragen und korrekten Antworten aus der CSV-Datei.
    """
    reference_data = {}
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            reference_data[row["Question"]] = row["Correct_Answer"]
    return reference_data

# Lade Fragen aus dem Fragenkatalog und Referenzantworten
reference_answers = load_reference_answers("reference_dataset.csv")

# Fragenkatalog
fragenkatalog = list(reference_answers.keys())

# Ergebnisberechnung und Konsolenausgabe
for frage in fragenkatalog:
    print(f"Question: {frage}")

    # KI-Antwort
    ki_antwort = query_llama_short(frage)
    print(f"KI Answer:\n{ki_antwort}")

    # Referenz-Antwort
    correct_answer = reference_answers[frage]
    print(f"Correct Answer:\n{correct_answer}")

    # Ähnlichkeitsprüfung
    similarity = calculate_semantic_similarity(ki_antwort, correct_answer)
    result = "Correct" if similarity > 0.7 else "Partially correct" if similarity > 0.4 else "Wrong"
    
    print(f"Similarity: {similarity:.2f}")
    print(f"Result: {result}")
    print("-" * 50)