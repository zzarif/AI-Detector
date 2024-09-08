import os
import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer, util


# process arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ft_model', type=str,
                    default='all-MiniLM-L6-v2', help='Specify the SBERT model')
args = parser.parse_args()
device = torch.device(
    f"cuda" if torch.cuda.is_available() else "cpu")


# Define evaluation metrics
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def dot_product(a, b):
    return np.dot(a, b)


# function to evaluate the fine-tuned model
def evaluate_model(model, df):
    # Encode all sentences
    candidate_embeddings = model.encode(df['candidate_combined'].tolist())
    ai_embeddings = model.encode(df['ai_combined'].tolist())

    # Calculate various similarity/distance measures
    cosine_scores = [cosine_similarity(c, a) for c, a in zip(
        candidate_embeddings, ai_embeddings)]
    manhattan_scores = [-manhattan_distance(c, a)
                        for c, a in zip(candidate_embeddings, ai_embeddings)]
    euclidean_scores = [-euclidean_distance(c, a)
                        for c, a in zip(candidate_embeddings, ai_embeddings)]
    dot_product_scores = [dot_product(c, a) for c, a in zip(
        candidate_embeddings, ai_embeddings)]

    true_scores = df['similarity_score'].tolist()

    # Calculate Spearman correlations
    cosine_spearman = spearmanr(true_scores, cosine_scores).correlation
    manhattan_spearman = spearmanr(true_scores, manhattan_scores).correlation
    euclidean_spearman = spearmanr(true_scores, euclidean_scores).correlation
    dot_product_spearman = spearmanr(
        true_scores, dot_product_scores).correlation

    # Calculate other metrics using cosine scores
    mse = mean_squared_error(true_scores, cosine_scores)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_scores, cosine_scores)
    r2 = r2_score(true_scores, cosine_scores)

    metrics = {
        "Cosine Spearman": cosine_spearman,
        "Manhattan Spearman": manhattan_spearman,
        "Euclidean Spearman": euclidean_spearman,
        "Dot Product Spearman": dot_product_spearman,
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
        "Mean Absolute Error": mae,
        "R-squared Score": r2
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    # Load the train data
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    df = pd.read_csv(os.path.join(data_dir, 'preprocessed_data.csv'))

    # Define model export/output path
    model_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, 'models')
    model_path = os.path.join(model_dir, f"fine-tuned_{args.ft_model}")

    model = SentenceTransformer(model_path, device=device)
    metrics = evaluate_model(model, df)
