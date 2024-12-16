from pymed import PubMed

# Verbindung zu PubMed herstellen
pubmed = PubMed()

fragenkatalog = [
    "What are the symptoms of diabetes?",
    "What is the treatment for hypertension?",
    "What are the side effects of ibuprofen?"
]

for frage in fragenkatalog:
    # Query an PubMed senden
    results = pubmed.query(frage, max_results=5)  # Maximal 5 Ergebnisse
    print(f"Results for '{frage}':")
    
    for article in results:
        # Titel und Abstract des Artikels ausgeben
        print(f"Title: {article.title}")
        print(f"Abstract: {article.abstract}")
        print("-" * 50)