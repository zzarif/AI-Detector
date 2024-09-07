import os
import argparse
import torch
from sentence_transformers import SentenceTransformer, util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='Specify GPU device for training')
parser.add_argument('--model', type=str,
                    default='all-MiniLM-L6-v2', help='Specify inference model')
args = parser.parse_args()
device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


def predict_similarity(model, question, candidate_answer, ai_answer):
    candidate_combined = f"Question: {question} Answer: {candidate_answer}"
    ai_combined = f"Question: {question} Answer: {ai_answer}"

    embeddings = model.encode([candidate_combined, ai_combined])
    similarity_score = util.pytorch_cos_sim(
        embeddings[0], embeddings[1]).item()

    return similarity_score


if __name__ == "__main__":
    # Load the trained model
    model_dir = data_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, 'models')
    model_path = os.path.join(model_dir, args.model)
    model = SentenceTransformer(model=model_path, device=device)

    # Example usage
    question = "Write a function to find the largest element in an array."
    candidate_answer = """
    def find_largest(arr):
        if not arr:
            return None
        max_element = arr[0]
        for element in arr[1:]:
            if element > max_element:
                max_element = element
        return max_element
    """
    ai_answer = """
    def find_largest(arr):
        return max(arr) if arr else None
    """
    
    similarity = predict_similarity(
        model, question, candidate_answer, ai_answer)
    print(f"Similarity score: {similarity:.4f}")
