from pymed import PubMed

def query_medicine(question):
    pubmed = PubMed(tool="AnswerEvaluation", email="youngleeee@outlook.com")
    results = pubmed.query(question, max_results=5)
    response_text = ""
    for article in results:
        response_text += f"{article.title}. "
    return response_text
