from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(api_response, reference_answer):
    embeddings1 = model.encode(api_response, convert_to_tensor=True)
    embeddings2 = model.encode(reference_answer, convert_to_tensor=True)
    similarity = util.cos_sim(embeddings1, embeddings2).item()
    return similarity
