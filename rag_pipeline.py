import os
import requests
import json
import time
import argparse
from dotenv import load_dotenv
from pathlib import Path
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import concurrent.futures

# Load environment variables
load_dotenv()

# Initialize embeddings for vector database
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Free open-source LLM API configurations
class LLMProvider:
    def __init__(self, name):
        self.name = name
    
    def query(self, prompt, mc_mode=False, max_retries=3, retry_delay=5):
        raise NotImplementedError("Subclasses must implement this method")

class HuggingFaceLLM(LLMProvider):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        super().__init__(f"huggingface-{model_name.split('/')[-1]}")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.api_token = os.getenv("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError("HF_API_TOKEN environment variable not set")
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def query(self, prompt, mc_mode=False, max_retries=3, retry_delay=2):
        """Query the Hugging Face model with retry mechanism"""
        retries = 0
        while retries <= max_retries:
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json={"inputs": prompt}
                )
                
                if response.status_code == 200:
                    api_response = response.json()
                    if api_response and isinstance(api_response, list) and "generated_text" in api_response[0]:
                        generated_text = api_response[0]["generated_text"].strip()
                        
                        # Remove the prompt from the beginning if present
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].strip()
                        
                        # For multiple choice, strictly extract only the letter
                        if mc_mode:
                            # Find the first occurrence of A, B, or C in the response
                            match = re.search(r'\b([A-C])\b', generated_text)
                            if match:
                                return match.group(1)  # Return just the letter
                            
                            # If no clear match, try to find any occurrence of A, B, or C
                            match = re.search(r'([A-C])', generated_text)
                            if match:
                                return match.group(1)  # Return just the letter
                            
                            # If still no match, return an error indicator
                            print(f"WARNING: Could not extract a valid answer (A, B, C) from: '{generated_text}'")
                            return "?"
                        
                        return generated_text
                    else:
                        print("Error: Invalid API response format.")
                        return "Error: Could not generate a valid response."
                elif response.status_code == 500 or "busy" in response.text.lower():
                    # This is where we handle Error 500 with retries
                    retries += 1
                    if retries <= max_retries:
                        print(f"Error 500: Model busy. Retrying ({retries}/{max_retries}) after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        # Increase delay for subsequent retries (exponential backoff)
                        retry_delay = min(retry_delay * 1.5, 10)
                    else:
                        print(f"Error: Failed after {max_retries} retries.")
                        return f"Error 500: Could not get a response from the model."
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    return f"Error {response.status_code}: Could not get a response from the model."
            except Exception as e:
                print(f"Exception during API call: {e}")
                return f"Error: {str(e)}"

# Keeping just the Hugging Face LLM implementation since we're using 3 Hugging Face models

def load_vector_database():
    """Load the vector database from disk."""
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vector_store

def retrieve_relevant_contexts(vector_store, question, k=3):
    """Retrieve the most relevant text passages for the question."""
    # Analyze the question
    analysis = analyze_question(question)
    
    # Get the k most similar contexts from the vector store
    documents = vector_store.similarity_search(
        question, 
        k=k
    )
    
    # Extract the text from the documents
    contexts = [doc.page_content for doc in documents]
    
    return contexts, analysis

def analyze_question(question):
    """Analyze the question to determine its intent and extract key entities."""
    # Simple keyword extraction
    keywords = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', question.lower())
    # Remove common stop words
    stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through', 'over', 'before', 'after', 'between', 'under', 'during', 'since', 'without'}
    keywords = [word for word in keywords if word not in stop_words and len(word) > 3]
    
    # Question classification (very simple approach)
    question_type = None
    if question.lower().startswith('what'):
        question_type = 'definition'
    elif question.lower().startswith('how'):
        question_type = 'procedure'
    elif question.lower().startswith('why'):
        question_type = 'explanation'
    elif question.lower().startswith('when'):
        question_type = 'time'
    elif question.lower().startswith('where'):
        question_type = 'location'
    elif question.lower().startswith('who'):
        question_type = 'person'
    else:
        question_type = 'general'
    
    return {
        'keywords': keywords,
        'question_type': question_type,
        'original_question': question
    }

def format_rag_prompt(question, contexts):
    """Format the RAG prompt by combining the question and retrieved contexts."""
    context_text = "\n\n".join([f"Context {i+1}:\n{context}" for i, context in enumerate(contexts)])
    
    prompt = f"""You are an educational assistant that helps answer questions based on course materials.
Use ONLY the following contexts from course materials to answer the question. If the answer is not contained in the contexts, say "I don't have enough information to answer this question based on the course materials."

{context_text}

Question: {question}

Answer: """
    
    return prompt

def format_mc_prompt(question, options):
    """Format a multiple-choice prompt for the model that strictly enforces a single-letter response."""
    options_text = "\n".join([f"{chr(65+i)}) {option}" for i, option in enumerate(options)])
    
    prompt = f"""You are answering a multiple-choice question. 

EXTREMELY IMPORTANT: Your ENTIRE response must consist of ONLY a SINGLE LETTER - either A, B, or C. 
Do not include ANY explanations, reasoning, punctuation, or additional text of ANY kind.
Do not include the parentheses or any other characters.
Invalid answer formats: "The answer is A", "A.", "(A)", "Option A", etc.
Valid answer format: just "A" or just "B" or just "C"

Question: {question}

Options:
{options_text}

Your answer (ONLY ONE LETTER, A, B, or C): """
    
    return prompt

def generate_answer(llm, question, contexts, analysis, answer_options=None):
    """Generate an answer based on the retrieved contexts using the specified LLM."""
    if answer_options:
        # For multiple choice questions
        mc_prompt = format_mc_prompt(question, answer_options)
        direct_mc_answer = llm.query(mc_prompt, mc_mode=True)
        
        # For detailed explanation (for logging purposes only)
        rag_prompt = format_rag_prompt(question, contexts)
        detailed_answer = llm.query(rag_prompt)
        
        return direct_mc_answer, detailed_answer
    else:
        # For open-ended questions
        prompt = format_rag_prompt(question, contexts)
        answer = llm.query(prompt)
        return answer, None

def process_with_llm(llm, question, vector_store, answer_options=None):
    """Process a question with a specific LLM."""
    # Retrieve relevant contexts
    contexts, analysis = retrieve_relevant_contexts(vector_store, question)
    
    # Generate answer
    answer, detailed_answer = generate_answer(llm, question, contexts, analysis, answer_options)
    
    return {
        'question': question,
        'answer': answer,
        'detailed_answer': detailed_answer,
        'contexts': contexts,
        'analysis': analysis
    }

def load_questions_from_json(json_file_path):
    """Load questions from a JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['fragenkatalog']

def evaluate_answers_by_difficulty(results, llm_name, output_file=None):
    """Evaluate the model's answers and calculate accuracy by difficulty level."""
    # Initialize counters for each difficulty level
    difficulty_counts = {
        "Leicht": {"correct": 0, "total": 0},
        "Mittel": {"correct": 0, "total": 0},
        "Schwer": {"correct": 0, "total": 0}
    }
    
    total_correct = 0
    total_questions = len(results)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(f"# Evaluation Results for {llm_name}\n\n")
            
            for i, result in enumerate(results, 1):
                # Get the difficulty level from the question metadata
                difficulty = result.get('difficulty', 'Unknown')
                
                file.write(f"## Question {i}: {result['question']}\n")
                file.write(f"**Difficulty:** {difficulty}\n")
                file.write(f"**Model Answer:** {result['answer']}\n")
                if result['detailed_answer']:
                    file.write(f"**Detailed Explanation:** {result['detailed_answer']}\n")
                file.write(f"**Correct Answer:** {result['correct_answer']} ({result['correct_answer_text']})\n")
                
                is_correct = result['answer'] == result['correct_answer']
                file.write(f"**Result:** {'✓ Correct' if is_correct else '✗ Incorrect'}\n\n")
                
                # Update counters
                if difficulty in difficulty_counts:
                    difficulty_counts[difficulty]["total"] += 1
                    if is_correct:
                        difficulty_counts[difficulty]["correct"] += 1
                        total_correct += 1
            
            # Calculate accuracy for each difficulty level
            file.write("## Summary by Difficulty Level\n\n")
            
            for difficulty, counts in difficulty_counts.items():
                if counts["total"] > 0:
                    accuracy = (counts["correct"] / counts["total"]) * 100
                    file.write(f"**{difficulty}:** {counts['correct']}/{counts['total']} ({accuracy:.2f}%)\n")
            
            # Overall accuracy
            overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
            file.write(f"\n**Overall:** {total_correct}/{total_questions} ({overall_accuracy:.2f}%)\n")
    
    # Return evaluation metrics
    evaluation = {
        'by_difficulty': difficulty_counts,
        'total': total_questions,
        'correct': total_correct,
        'accuracy': (total_correct / total_questions) * 100 if total_questions > 0 else 0
    }
    
    return evaluation

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline with multiple LLM evaluations")
    parser.add_argument("--json", default="Fragenkatalog.json", help="Path to the JSON file with questions")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of questions to process")
    args = parser.parse_args()
    
    # Check if vector database exists
    if not Path("faiss_index").exists():
        print("Error: Vector database not found. Please run the original script to build it first.")
        return
    
    # Load vector database
    print("Loading vector database...")
    vector_store = load_vector_database()
    
    # Load questions from JSON file
    json_file_path = args.json
    print(f"Loading questions from {json_file_path}...")
    questions = load_questions_from_json(json_file_path)
    
    # Limit questions if specified
    if args.limit:
        questions = questions[:args.limit]
        print(f"Limited to processing {args.limit} questions")
    
    # Initialize LLM providers
    try:
        llm_providers = [
            #HuggingFaceLLM("meta-llama/Meta-Llama-3-8B-Instruct"),   # Original model from your code
            HuggingFaceLLM("microsoft/phi-2"),                        # Microsoft's Phi-2 model
            HuggingFaceLLM("google/flan-t5-large")                    # Google's Flan-T5 model
        ]
        
        print(f"Initialized {len(llm_providers)} LLM providers")
        for llm in llm_providers:
            print(f"  - {llm.name}")
    except ValueError as e:
        print(f"Error initializing LLM providers: {e}")
        print("Please set required API keys in .env file.")
        return
    
    # Process questions with each LLM provider
    for llm in llm_providers:
        results = []
        print(f"\nProcessing {len(questions)} questions with {llm.name}...")
        
        for i, q in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {q['id']} with {llm.name}")
            question_text = q['frage']
            options = q['antwortmoeglichkeiten']
            correct_answer_text = q['korrekte_antwort']
            difficulty = q.get('schwierigkeit', 'Unknown')
            
            # Find the index of the correct answer to determine the letter (A, B, C)
            correct_index = options.index(correct_answer_text)
            correct_answer = chr(65 + correct_index)  # A, B, C, etc.
            
            # Use the RAG pipeline to answer the question
            result = process_with_llm(llm, question_text, vector_store, options)
            
            # Add the correct answer information and difficulty
            result['correct_answer'] = correct_answer
            result['correct_answer_text'] = correct_answer_text
            result['difficulty'] = difficulty
            
            # Print progress
            print(f"  Model answer: {result['answer']}, Correct: {correct_answer}, Difficulty: {difficulty}")
            results.append(result)
        
        # Evaluate the results by difficulty
        output_file = f"evaluation_results_{llm.name}.txt"
        print(f"\nEvaluating results for {llm.name}...")
        evaluation = evaluate_answers_by_difficulty(results, llm.name, output_file)
        
        # Print summary
        print(f"\nEvaluation Summary for {llm.name}:")
        for difficulty, counts in evaluation['by_difficulty'].items():
            if counts["total"] > 0:
                accuracy = (counts["correct"] / counts["total"]) * 100
                print(f"{difficulty}: {counts['correct']}/{counts['total']} ({accuracy:.2f}%)")
        
        print(f"Overall: {evaluation['correct']}/{evaluation['total']} ({evaluation['accuracy']:.2f}%)")
        print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()