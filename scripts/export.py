import torch
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse
import os


# process arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ft_model', type=str,
                    default='all-MiniLM-L6-v2', help='Specify inference model')
args = parser.parse_args()
device = torch.device(
    f"cuda" if torch.cuda.is_available() else "cpu")


# funtion to convert fine-tuned model to ONNX
def export_to_onnx(model_path, onnx_path):
    # Load the SentenceTransformer model
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare dummy input (adjust input size as needed)
    inputs = tokenizer("Candidate coding answer", "AI generated answer",
                       padding=True, truncation=True, return_tensors="pt")

    # Export the model
    torch.onnx.export(
        model,
        # Model input (tuple)
        args=(inputs['input_ids'], inputs['attention_mask']),
        f=onnx_path,  # Output ONNX file path
        # Names of the model inputs
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],  # Name of the model output
        dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {
            # For dynamic input shapes
            0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )

    print(f"Model exported to {onnx_path}")


# Model quantization
def quantize_onnx_model(onnx_path, quantized_onnx_path):
    quantize_dynamic(onnx_path,
                     quantized_onnx_path,
                     weight_type=QuantType.QUInt8)  # 8-bit integers for weights
    print(f"Quantized model saved to: {quantized_onnx_path}")


if __name__ == "__main__":
    # Load the model
    model_dir = data_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, 'models')
    model_path = os.path.join(model_dir, f"fine-tuned_{args.ft_model}")
    onnx_path = os.path.join(model_dir, f"fine-tuned_{args.ft_model}.onnx")

    # Export the model
    export_to_onnx(model_path, onnx_path)

    # compress the model
    quantize_onnx_path = os.path.join(
        model_dir, f"fine-tuned_{args.ft_model}_quantized.onnx")
    quantize_onnx_model(onnx_path, quantize_onnx_path)
