from config import chain
import os


# Function to load text from a file
def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


# Function to get similarity score from LLM
def get_similarity_score(question, candidate_answer, ai_answer):
    # Run the LLM chain with the given inputs
    response = chain.invoke({
        "question": question,
        "candidate_answer": candidate_answer,
        "ai_answer": ai_answer
    })

    # Parse and return the similarity score
    try:
        score = float(response.content.strip())
        return score
    except ValueError:
        raise ValueError("The model did not return a valid score.")


# Example usage
if __name__ == "__main__":
    # Load content from files
    question = load_text_from_file(os.path.join(
        os.path.dirname(__file__), 'question.txt'))
    candidate_answer = load_text_from_file(os.path.join(
        os.path.dirname(__file__), 'candidate_answer.txt'))
    ai_answer = load_text_from_file(os.path.join(
        os.path.dirname(__file__), 'ai_answer.txt'))

    similarity_score = get_similarity_score(
        question, candidate_answer, ai_answer)
    print(f"Similarity Score: {similarity_score}")
