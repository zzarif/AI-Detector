import os
import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='all-MiniLM-L6-v2', help='Specify the SBERT model')
parser.add_argument('--gpu', type=int, default=0,
                    help='Specify GPU device for training')
args = parser.parse_args()
device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


def train_model(df, model_name, output_path, epochs=5, batch_size=16):
    # Split the data into train and test sets
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create examples for training
    train_examples = [InputExample(texts=[row['candidate_combined'], row['ai_combined']], label=float(
        row['similarity_score'])) for _, row in train_df.iterrows()]

    # Create DataLoader for training with appropriate batch size
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size)

    # Initialize the specified SentenceTransformer model
    model = SentenceTransformer(model=model_name, device=device)

    # Define the appropriate loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Prepare validation data
    valid_samples = [(row['candidate_combined'], row['ai_combined'], row['similarity_score'])
                     for _, row in valid_df.iterrows()]
    valid_examples = [InputExample(
        texts=[s[0], s[1]], label=float(s[2])) for s in valid_samples]

    # Create an evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        valid_examples, name='validation')

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=epochs,
              warmup_steps=100,
              evaluator=evaluator,
              evaluation_steps=500,
              output_path=output_path)


if __name__ == "__main__":
    # Load the train data
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    df = pd.read_csv(os.path.join(data_dir, 'preprocessed_data.csv'))

    # Define model export/output path
    model_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, 'models')
    output_path = os.path.join(model_dir, args.model)

    # Train the model
    train_model(df, args.model, output_path)

    print("Model training complete.")
