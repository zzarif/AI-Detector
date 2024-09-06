import os
import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str,
                    default='preprocessed_data.csv', help='Specify the preprocessed dataset')
parser.add_argument('--model', type=str,
                    default='all-MiniLM-L6-v2', help='Specify the SBERT model')
parser.add_argument('--gpu', type=int, default=0,
                    help='Specify GPU device for training')
args = parser.parse_args()
device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Load the preprocessed data
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    df = pd.read_csv(os.path.join(data_dir, args.train_data))

    # Split the data into train and test sets
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create examples for training
    train_examples = [InputExample(texts=[row['candidate_combined'], row['ai_combined']], label=float(
        row['similarity_score'])) for _, row in train_df.iterrows()]

    # Create DataLoader for training with appropriate batch size
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Initialize the specified SentenceTransformer model
    model = SentenceTransformer(model=args.model, device=device)

    # Define the appropriate loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Define evaluation params
    candidate_combined = valid_df['candidate_combined'].tolist()
    ai_combined = valid_df['ai_combined'].tolist()
    similarity_score = valid_df['similarity_score'].tolist()

    # Define evaluator for embeddings
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        candidate_combined, ai_combined, similarity_score)

    # Define model export/output path
    model_dir = data_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, 'models')
    output_path = os.path.join(model_dir, args.model)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=5,
              warmup_steps=100,
              evaluator=evaluator,
              evaluation_steps=500,
              output_path=output_path)

    print("Model training complete.")
