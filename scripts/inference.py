import os
import argparse
import torch
from sentence_transformers import SentenceTransformer, util


parser = argparse.ArgumentParser()
parser.add_argument('--test_data', type=str,
                    default='test_data.csv', help='Specify test data')
parser.add_argument('--gpu', type=int, default=0,
                    help='Specify GPU device for training')
parser.add_argument('--model', type=str,
                    default='paraphrase-MiniLM-L6-v2', help='Specify inference model')
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
    question = "What is the time complexity of quicksort?"
    candidate_answer = "The average time complexity of quicksort is O(n log n)."
    ai_answer = "Quicksort has an average and best-case time complexity of O(n log n), but in the worst case, it can be O(n^2)."

    similarity = predict_similarity(
        model, question, candidate_answer, ai_answer)
    print(f"Similarity score: {similarity:.4f}")
