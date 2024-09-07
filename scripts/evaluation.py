import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer, util


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='all-MiniLM-L6-v2', help='Specify the SBERT model')
parser.add_argument('--gpu', type=int, default=0,
                    help='Specify GPU device for training')
args = parser.parse_args()
device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, df):
    # Encode all sentences
    candidate_embeddings = model.encode(df['candidate_combined'].tolist())
    ai_embeddings = model.encode(df['ai_combined'].tolist())

    # Calculate cosine similarities
    cosine_scores = [util.pytorch_cos_sim(c, a).item(
    ) for c, a in zip(candidate_embeddings, ai_embeddings)]

    # Calculate evaluation metrics
    mse = mean_squared_error(df['similarity_score'], cosine_scores)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df['similarity_score'], cosine_scores)
    r2 = r2_score(df['similarity_score'], cosine_scores)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    return mse, rmse, mae, r2


if __name__ == "__main__":
    # Load the train data
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))

    # Define model export/output path
    model_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, 'models')
    model_path = os.path.join(model_dir, args.model)

    model = SentenceTransformer(model=model_path, device=device)
    mse, rmse, mae, r2 = evaluate_model(model, df)
