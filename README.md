# cs613_-text_summarization

# Text Summarization and Evaluation

This project demonstrates various text summarization techniques (TF-IDF, TextRank, T5) and evaluates their performance using ROUGE scores on the CNN/Daily Mail and Amazon Reviews datasets.

## Prerequisites

Ensure you have Python installed (version 3.6 or higher). You'll also need to install the following libraries:

- `nltk`
- `numpy`
- `networkx`
- `matplotlib`
- `scikit-learn`
- `transformers`
- `datasets`
- `rouge-score`

You can install these libraries using pip:

```bash
pip install nltk numpy networkx matplotlib scikit-learn transformers datasets rouge-score
```

### Download NLTK Data

Download the necessary NLTK data:
import nltk
nltk.download('punkt')

### How to Run the Code
	1	Load and Preprocess Data: The script loads the CNN/Daily Mail and Amazon Reviews datasets and preprocesses the texts.
	2	Summarization Methods:
	◦	TF-IDF Summarization: Summarizes the text based on TF-IDF scores of sentences.
	◦	TextRank Summarization: Summarizes the text using the TextRank algorithm.
	◦	T5 Summarization: Uses the T5 model to generate summaries.
	3	Generate Summaries: The script generates summaries for example texts from both datasets using each of the summarization methods.
	4	Plotting the Summaries: Summaries are displayed alongside the original text using Matplotlib.
	5	Evaluate Summaries with ROUGE: The script evaluates the generated summaries against the original summaries in the CNN/Daily Mail dataset using ROUGE scores.
	6	Plot ROUGE Scores: The average ROUGE scores for each summarization method are plotted for comparison.
	7	Print Average ROUGE Scores: The script prints the average ROUGE scores for each summarization method.
Running the Script
Simply execute the script in your Python environment:
```bash
Python cs613_final.py
```bash

