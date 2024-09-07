import os
import json
import pandas as pd


file_extensions_dict = {
    "Java": "jav",
    "JavaScript": "js",
    "C++": "cpp",
    "Python": "py",
    "Swift": "swift",
    "React": "js",
    "Kotlin": "kt",
    "SQL": "sql",
    "Docker": "dockerfile",
    "C#": "cs",
    "PHP & JavaScript": "js",
    "HTML/CSS": "css",
    "JavaScript-React": "js",
    "PHP": "php",
    "TypeScript": "ts",
    "Go": "go",
    "Dart": "dart"
}


# Function to combine question and answer
def combine_question_answer(question, answer):
    return f"Question: {question} Answer: {answer}"


def load_question_data(source_codes_path):
    questions = []
    for folder in os.listdir(source_codes_path):
        if folder.startswith('source_code_'):
            question_path = os.path.join(source_codes_path, folder)
            question_data = {}

            # Load and save JSON question data
            with open(os.path.join(question_path, f'{folder}.json'), 'r') as f:
                metadata = json.load(f)
                question_data['id'] = folder
                question_data['question'] = metadata['question']

            # Load candidate's answer
            with open(os.path.join(question_path, f'{folder}.{file_extensions_dict.get(metadata["programming_language"], '')}'), 'r') as f:
                question_data['candidate_answer'] = f.read()

            # Load AI-generated answers
            ai_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
            for model in ai_models:
                for i in range(2):
                    with open(os.path.join(question_path, f'{folder}_{model}_0{i}.{file_extensions_dict.get(metadata["programming_language"], '')}'), 'r') as f:
                        question_data[f'{model}_0{i}'] = f.read()

            questions.append(question_data)

    return questions


def preprocess_data(source_codes_path, source_codes_labels):
    questions = load_question_data(source_codes_path)
    plagiarism_scores = pd.read_excel(source_codes_labels)

    data = []
    for question in questions:
        for ai_model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']:
            for i in range(2):
                score = plagiarism_scores[
                    (plagiarism_scores['coding_problem_id'] == question['id']) &
                    (plagiarism_scores['llm_answer_id'] == f'{ai_model}_0{i}')
                ]['plagiarism_score'].values[0]

                data.append({
                    'question': question['question'],
                    'candidate_answer': question['candidate_answer'],
                    'ai_answer': question[f'{ai_model}_0{i}'],
                    'similarity_score': score
                })

    df = pd.DataFrame(data)

    df['candidate_combined'] = df.apply(lambda row: combine_question_answer(
        row['question'], row['candidate_answer']), axis=1)
    df['ai_combined'] = df.apply(lambda row: combine_question_answer(
        row['question'], row['ai_answer']), axis=1)

    return df


if __name__ == "__main__":
    # Load the original data
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    source_codes_path = os.path.join(data_dir, 'dataset-source-codes')
    source_codes_labels = os.path.join(
        data_dir, 'CodeAid Source Codes Labeling.xlsx')

    # Preprocess the data
    df = preprocess_data(source_codes_path, source_codes_labels)

    # Save the preprocessed data
    preprocessed_data = os.path.join(data_dir, 'preprocessed_data.csv')
    df.to_csv(preprocessed_data, index=False)
    print("Data preprocessing complete.")
