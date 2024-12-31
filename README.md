# Misinformation Detection Tool

## Overview

Misinformation has become a significant issue in today's digital age, influencing public opinion and spreading unreliable news. This project addresses the problem by building a robust **Misinformation Detecting Tool** using advanced **Hugging Face Transformers**. The system is capable of identifying whether a given news article or statement is genuine or fake.

## Problem Statement

The rapid spread of misinformation through online platforms and social media has created the need for reliable tools to combat fake news. Identifying fake news manually is time-consuming and prone to bias. This project automates the detection of fake news using natural language processing (NLP) techniques, ensuring scalability and accuracy.

## Objective

The objective of this project is to develop and deploy a machine learning model capable of analyzing textual data and accurately classifying it as either **real** or **fake** news. The solution is deployed using Hugging Face Transformers to make it accessible and scalable.

## Features

- **Deep Learning Model**: Built on Hugging Face Transformers for state-of-the-art text classification.
- **Scalable Deployment**: Deployed on Hugging Face for seamless integration and access.
- **Real-Time Prediction**: Provides instant results for news articles or headlines.

## Methodology

1. **Data Collection**:
   - Collected datasets from Kaggle and other reliable sources containing labeled news articles.
2. **Data Preprocessing**:

   - Cleaned and tokenized text data.
   - Removed stop words, special characters, and performed lemmatization.

3. **Model Selection**:

   - Used a pre-trained transformer model (e.g., BERT, RoBERTa) from Hugging Face.
   - Fine-tuned the model on the fake news dataset.

4. **Training**:

   - Split the dataset into training and validation sets.
   - Used PyTorch backend for training with optimization techniques.

5. **Evaluation**:

   - Measured performance using metrics like accuracy, precision, recall, and F1-score.
   - Validated the model with a test dataset to ensure generalizability.

6. **Deployment**:
   - Deployed the model on Hugging Face for public access.
   - API created for real-time predictions.

## Scope

- **Immediate Use**: Detects fake news effectively from textual inputs such as headline or article links.
- **Future Enhancements**:
  - Incorporating language detection and translation for multilingual support.
  - Extending the dataset to include more diverse topics and sources.
  - Integration with video and audio analysis for multimedia content.
  - Expanded database for fact-checking and knowledge graphs.

## Installation and Usage

### Local Setup

1. Clone the repository:

   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/malavika4089/misinformation-truthteller/tree/main
   cd misinformation-truthteller
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   streamlit run app.py
   ```

### Access Deployed Model

The model is deployed on Hugging Face. You can access it [Live link](https://huggingface.co/spaces/malavika4089/misinformation-truthteller).

## Dataset

The dataset used for this project was sourced from:

- [Kaggle Fake And Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Hugging Face Transformers, PyTorch, Scikit-learn, NumPy, Pandas, streamlit
- **Deployment**: Hugging Face Spaces,
- **Tools**: Colab

## License

This project is licensed under the [MIT License](LICENSE).

