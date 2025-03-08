# RAG-System Anleitung

## Setup

1. **Pakete installieren**:
   ```bash
   pip install langchain langchain-community faiss-cpu sentence-transformers python-dotenv pypdf2 python-docx python-pptx requests
   ```

2. **API-Token einrichten**:
   `.env`-Datei erstellen:
   ```
   HF_API_TOKEN=Dein_Hugging_Face_Token
   ````

4. **Hugging Face Zugang**:
   - Account erstellen
   - API-Token generieren
   - Llama-3-Modell Zugang beantragen

## Ausführung

1. Kursmaterialien in "course_materials" ablegen
3. Ausführen:
   ```bash
   python rag_pipeline.py
   ```

Erster Start dauert länger (Datenbank-Aufbau), spätere Starts sind schneller.

## Methoden
- Verarbeitet automatisch deine Kursmaterialien
- Baut eine Vektordatenbank 
- Startet eine Frage-Antwort-Session
- analysiert Halluzinationen

Viel Erfolg mit dem Projekt!