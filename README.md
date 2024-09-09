<h1 align="center">
  <br>
  AI Detector - Deployment
  <br>
</h1>

<h4 align="center">Detect AI generated coding answers</h4>

<p align="center">
  <a href="https://github.com/zzarif/AI-Detector">
    <img src="https://img.shields.io/github/last-commit/zzarif/AI-Detector">
  </a>
  <a href="https://multilabel-scifi-tags-classifier.vercel.app">
    <img src="https://img.shields.io/badge/live-vercel-red.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-yellow.svg">
  </a>
</p>

# Table of Contents

<ul>
  <li><a href="#problem-statement">Problem Statement</a></li>
  <li><a href="#flask-web-deployment">Flask Web Deployment</a></li>
  <li><a href="#build-from-source">Build from Source</a></li>
  <li><a href="#contact">Contact</a></li>
</ul>

# Problem Statement

The objective of this project is to develop a Machine Learning model that can detect potential AI use by comparing candidate coding answers to responses generated by AI models (GPT-4, GPT-4 Turbo, and GPT-3.5 Turbo). The model will predict AI-detected score ranging in floating point numbers from `0` (no AI detected) to `1` (a lot of AI-detected). Since the prediction is a continuous value in a range, this is a **Regression** problem which can be solved in multiple ways. In this problem's context, I demonstrated **two** solutions.

The first solution where I fine-tune multiple Sentence Transformers on the given dataset, I deployed the best performing fine-tuned model to [HuggingFace](https://huggingface.co/spaces/zzarif/AI-Detector) and integrated with a [Flask Webapp](https://ai-detector-scopic.vercel.app/).

## Flask Web Deployment

A custom [Web Application](https://ai-detector-scopic.vercel.app/) was developed with Flask to demonstrate the AI detection model's capability. It uses the HuggingFace Spaces API in the backend. Following are some snapshots of the Flask webapp:

[attach image]

## Build from Source

1. Clone the repo

```bash
git clone https://github.com/zzarif/AI-Detector.git
cd AI-Detector/
git switch flask
```

2. Initialize and activate virtual environment

```bash
virtualenv venv
source venv/Scripts/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

_Note: Select virtual environment interpreter from_ `Ctrl`+`Shift`+`P`

4. Run the script

```bash
python app.py
```

## Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/zibran-zarif-amio-b82717263/) [![Mail](https://img.shields.io/badge/Gmail-EA4335?logo=gmail&logoColor=fff)](mailto:zibran.zarif.amio@gmail.com)
