import gradio as gr
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load the ONNX model
onnx_model_path = "fine-tuned_all-distilroberta-v1_quantized.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-distilroberta-v1")


def predict_similarity(question, candidate_answer, ai_answer):
    # Combine question and answers
    candidate_combined = f"Question: {question} Answer: {candidate_answer}"
    ai_combined = f"Question: {question} Answer: {ai_answer}"

    # Tokenize inputs
    inputs = tokenizer([candidate_combined, ai_combined],
                       padding=True, truncation=True, return_tensors="np")

    # Run inference
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    # Get embeddings (shape: (seq_length, 768))
    embeddings = ort_outputs[0]

    # Apply mean pooling to reduce (seq_length, 768) to (768,)
    candidate_embedding = np.mean(embeddings[0], axis=0)  # Shape (768,)
    ai_embedding = np.mean(embeddings[1], axis=0)  # Shape (768,)

    # Calculate cosine similarity
    similarity = np.dot(candidate_embedding, ai_embedding) / \
        (np.linalg.norm(candidate_embedding) * np.linalg.norm(ai_embedding))

    return float(similarity)


# Create Gradio interface
iface = gr.Interface(
    fn=predict_similarity,
    inputs=[
        gr.Textbox(label="Coding Question"),
        gr.Textbox(label="Candidate's Answer"),
        gr.Textbox(label="AI-generated Answer")
    ],
    outputs=gr.Number(label="Similarity Score"),
    title="AI Code Detector",
    description="Detect similarity between human-written and AI-generated coding answers."
)

# Launch the app
iface.launch()
