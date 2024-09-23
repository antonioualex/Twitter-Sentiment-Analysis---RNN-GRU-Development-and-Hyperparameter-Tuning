# Twitter Sentiment Analysis - RNN-GRU Development

## Overview
This project focuses on the development of a Recurrent Neural Network (RNN) model using Gated Recurrent Units (GRUs) for
Twitter sentiment analysis, along with hyperparameter tuning. The performance of the RNN-GRU model is compared against two pre-existing models:
- [Logistic Regression Model (Hyperparameter Tuned)](https://github.com/antonioualex/Tweet-Sentiment-Classifier-with-Machine-Learning-Models)
- [MLP Model (Hyperparameter Tuned)](https://github.com/antonioualex/Tweet-Sentiment-Classifier-using-MLPs)

## Dataset

The dataset used is Kaggle's [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). 
It originally contains 100,000 tweets with unbalanced sentiment labels. For this project, we balanced the dataset 
by selecting 5,000 positive and 5,000 negative samples to ensure fairness in model evaluation.

## Project Objectives

The key objectives of the project are:
1. **Data Processing**: Preprocessing the Kaggle Twitter Sentiment Analysis Dataset by balancing the dataset, 
tokenizing, lemmatizing, and applying special token replacements. More specifically:
   - **Tokenization & Lemmatization**: Tokenized the text and applied lemmatization using the NLTK library to reduce dimensionality. POS tags were used to ensure correct lemmatization.
   - **Special Token Replacement**: Replaced URLs, usernames, hashtags, numbers, emojis, and emoticons with placeholders to prevent them from skewing sentiment analysis.
   - **Lowercasing**: Converted all text to lowercase for uniformity.
2. **Feature Engineering**: Representing text data using TF-IDF and Word Embeddings, with 
experimentation on centroid-based embeddings and recurrent networks. More specifically:
   - **TF-IDF Vectorizer**: Using `TfidfVectorizer`, unigram and bigram features were extracted. The vectorizer was
      limited to a maximum of 5000 features and stop words were removed.
   - **Word Embeddings**: Pre-trained GloVe embeddings (Twitter-based, 100 dimensions) were used to convert the text 
      into dense vectors, capturing semantic relationships.

3. **Model Development**:
   - Dummy Classifier (Baseline)
   - Logistic Regression with Cross-Validation
   - Multi-Layer Perceptron (MLP) with TF-IDF and Word Embedding features.
   - Recurrent Neural Network (RNN) with BiGRU architecture.
4. **Hyperparameter Tuning**: Using Keras Tuner for optimizing both MLP and RNN models.
5. **Model Evaluation**: Evaluating performance with learning curves, precision-recall curves, and F1-score metrics.



## Baseline Models

#### Dummy Classifier
We start with a baseline using a dummy classifier that always predicts the most frequent class in the training set.
This acts as a benchmark for model comparison.

#### Logistic Regression (Hyperparameter Tuned)
The logistic regression model was hyperparameter-tuned. 
[Link to Model](https://github.com/example/logistic-regression).

#### Multi-Layer Perceptron (MLP) Model (Hyperparameter Tuned)
The MLP model uses pre-trained GloVe embeddings (`glove-twitter-100`) and underwent hyperparameter tuning using the
Keras Tuner library. [Link to Model](https://github.com/example/mlp-model).

## RNN-GRU Model Development

We developed a BiGRU (Bidirectional GRU) architecture to handle sequential data from both forward and backward directions.
Key features of the model include:
   - **Text Preprocessing**: Tweets were pre-processed using the Spacy tagger for tokenization and cleaning.
  - **Embedding Layer**: We created an embedding matrix using GloVe to transform tokenized sequences into dense vectors.
  - **GRU Layers**: BiGRU with Gated Recurrent Units was employed to handle long-term dependencies.
  - **Self-Attention Layer**: Integrated a Deep Self-Attention layer to focus on key parts of the sequence.

#### Hyperparameter Tuning

The tuning process involved a random search strategy with a maximum of 10 trials. 
The objective was to maximize the validation categorical accuracy. Early stopping was used during the search to 
prevent overfitting and save time and resources.

The hyperparameters being tuned included:
- Dropout rate for the Dropout layers
- Number of RNN layers
- Number of units in the GRU and Dense layers
- Number of layers and units in the Self-Attention layer
- Learning rate for the Adam optimizer

The final tuned model architecture included a self-attention layer for enhanced performance.

## Evaluation

The models were evaluated using the following techniques:
- **Learning Curves**: Accuracy and loss were tracked across epochs for both MLP and BiGRU models.
- **Precision-Recall Curves**: Evaluated the trade-off between precision and recall for all models, 
with Area Under the Curve (AUC-PR) as a key metric.
- **F1 Score**: Both macro-averaged and weighted-averaged F1 scores were calculated to ensure fair evaluation 
across imbalanced classes.
