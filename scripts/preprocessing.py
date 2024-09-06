import os
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='data.csv', help='Specify the dataset')
args = parser.parse_args()


# Function to combine question and answer
def combine_question_answer(row):
    return f"Question: {row['question']} Answer: {row['answer']}"


if __name__ == "__main__":
    # Load the original data
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    df = pd.read_csv(os.path.join(data_dir, args.dataset))

    # Combine coding question with candidate and AI answers and add corresponding columns
    df['candidate_combined'] = df.apply(lambda row: combine_question_answer(
        {'question': row['question'], 'answer': row['candidate_answer']}), axis=1)
    df['ai_combined'] = df.apply(lambda row: combine_question_answer(
        {'question': row['question'], 'answer': row['ai_answer']}), axis=1)

    preprocessed_data = os.path.join(data_dir, 'preprocessed_data.csv')
    df.to_csv(preprocessed_data, index=False)

    print("Data preprocessing complete.")
