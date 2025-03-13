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
import random
from ctransformers import AutoModelForCausalLM, Config

# Load environment variables
load_dotenv()

# Initialize embeddings for vector database
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class LLMProvider:
    def __init__(self, name):
        self.name = name
    
    def query(self, prompt, mc_mode=False, max_retries=3, retry_delay=5):
        raise NotImplementedError("Subclasses must implement this method")

class LocalLLM(LLMProvider):
    def __init__(self, model_name, model_path):
        super().__init__(model_name)
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please make sure the model file exists.")
            
        # Configure for AMD GPU using ROCm
        os.environ["HIP_VISIBLE_DEVICES"] = "0"  # Use the first GPU
        os.environ["GPU_MAX_HEAP_SIZE"] = "90%"  # Allow using most of GPU memory
        os.environ["ROCM_PATH"] = "C:\\Program Files\\AMD\\ROCm"  # Updated ROCm installation path
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral" if "mistral" in model_name else "llama",
            max_new_tokens=512,
            temperature=0.5,
            context_length=2048,
            gpu_layers=35,  # Number of layers to offload to GPU
            batch_size=1,   # Reduced batch size for GPU memory
            threads=4,
            local_files_only=True  # Force using local files only
        )
    
    def query(self, prompt, mc_mode=False, max_retries=3, retry_delay=5):
        """Query the local model with retry mechanism"""
        retries = 0
        while retries <= max_retries:
            try:
                # Generate response
                generated_text = self.llm(prompt, stop=["Question:", "\n\n"], max_new_tokens=512)
                
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
                
                return generated_text.strip()
                
            except Exception as e:
                print(f"Exception during model inference: {e}")
                retries += 1
                if retries <= max_retries:
                    print(f"Retrying ({retries}/{max_retries}) after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return f"Error: {str(e)}"

class MistralLLM(LocalLLM):
    def __init__(self):
        super().__init__("mistral-7b", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

class DeepSeekLLM(LocalLLM):
    def __init__(self):
        super().__init__("deepseek-llm-7b", "models/deepseek-llm-7b-chat.Q4_K_M.gguf")

def load_vector_database():
    """Load the vector database from disk."""
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vector_store

def retrieve_relevant_contexts(vector_store, question, k=3):
    """Retrieve the most relevant text passages for the question."""
    # Get the k most similar contexts from the vector store
    documents = vector_store.similarity_search(
        question, 
        k=k
    )
    
    # Extract the text from the documents
    contexts = [doc.page_content for doc in documents]
    
    return contexts

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

def generate_answer(llm, question, contexts, answer_options=None):
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
    contexts = retrieve_relevant_contexts(vector_store, question)
    
    # Generate answer
    answer, detailed_answer = generate_answer(llm, question, contexts, answer_options)
    
    return {
        'question': question,
        'answer': answer,
        'detailed_answer': detailed_answer,
        'contexts': contexts
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
    parser = argparse.ArgumentParser(description="RAG Pipeline for Question Answering")
    parser.add_argument("--question", type=str, required=False, help="Optional: Single question to answer. If not provided, will use questions from Fragenkatalog.json")
    parser.add_argument("--mc", action="store_true", help="Run in multiple choice mode")
    parser.add_argument("--options", nargs="+", help="Multiple choice options")
    parser.add_argument("--llm", type=str, choices=["mistral", "deepseek"], 
                      default="mistral", help="LLM to use")
    args = parser.parse_args()

    # Initialize the appropriate LLM
    if args.llm == "mistral":
        llm = MistralLLM()
    elif args.llm == "deepseek":
        llm = DeepSeekLLM()
    else:
        raise ValueError(f"Unknown LLM: {args.llm}")

    # Check if vector database exists
    if not Path("faiss_index").exists():
        print("Error: Vector database not found. Please run the original script to build it first.")
        return
    
    # Load vector database
    print("Loading vector database...")
    vector_store = load_vector_database()
    
    # Load questions from JSON file
    json_file_path = "Fragenkatalog.json"
    print(f"Loading questions from {json_file_path}...")
    questions = load_questions_from_json(json_file_path)
    
    # Process questions with the selected LLM
    results = []
    print(f"\nProcessing {len(questions)} questions with {llm.name}...")
    
    for i, q in enumerate(questions, 1):
        try:
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
            
            # Save progress after each question
            output_file = f"evaluation_results_{llm.name}.txt"
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(f"# Evaluation Results for {llm.name}\n\n")
                for j, r in enumerate(results, 1):
                    file.write(f"## Question {j}: {r['question']}\n")
                    file.write(f"**Difficulty:** {r['difficulty']}\n")
                    file.write(f"**Model Answer:** {r['answer']}\n")
                    if r['detailed_answer']:
                        file.write(f"**Detailed Explanation:** {r['detailed_answer']}\n")
                    file.write(f"**Correct Answer:** {r['correct_answer']} ({r['correct_answer_text']})\n")
                    file.write(f"**Result:** {'✓ Correct' if r['answer'] == r['correct_answer'] else '✗ Incorrect'}\n\n")
                
        except KeyboardInterrupt:
            print("\nScript interrupted by user. Saving progress...")
            break
        except Exception as e:
            print(f"\nError processing question {i}: {str(e)}")
            print("Saving progress and continuing with next question...")
            continue
    
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