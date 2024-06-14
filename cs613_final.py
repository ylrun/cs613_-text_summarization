import nltk
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from rouge import Rouge

nltk.download('punkt')

# Load the CNN/Daily Mail and Amazon Reviews datasets
cnn_dataset = load_dataset("cnn_dailymail", '3.0.0', split='test[:1%]')
amazon_dataset = load_dataset("amazon_polarity", split='test[:1%]')

# Preprocess the datasets
def preprocess_data(dataset, text_key, summary_key=None):
    texts = []
    summaries = []
    for data in dataset:
        text = data[text_key]
        texts.append(text)
        if summary_key:
            summary = data[summary_key]
            summaries.append(summary)
    return texts, summaries

cnn_texts, cnn_summaries = preprocess_data(cnn_dataset, 'article', 'highlights')
amazon_texts, _ = preprocess_data(amazon_dataset, 'content')

# Summarization methods
def tfidf_summarizer(text, num_sentences=5):
    sentences = sent_tokenize(text)
    if len(sentences) < num_sentences:
        return ' '.join(sentences)  # Return the entire text if fewer sentences than num_sentences
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[-num_sentences:]]
    return ' '.join(ranked_sentences)

def textrank_summarizer(text, num_sentences=5):
    sentences = sent_tokenize(text)
    if len(sentences) < num_sentences:
        return ' '.join(sentences)  # Return the entire text if fewer sentences than num_sentences
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

def t5_summarizer(text):
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Generate summaries for the first text in each dataset
cnn_example = cnn_texts[0]
tfidf_summary_cnn = tfidf_summarizer(cnn_example)
textrank_summary_cnn = textrank_summarizer(cnn_example)
t5_summary_cnn = t5_summarizer(cnn_example)

amazon_example = amazon_texts[0]
tfidf_summary_amazon = tfidf_summarizer(amazon_example)
textrank_summary_amazon = textrank_summarizer(amazon_example)
t5_summary_amazon = t5_summarizer(amazon_example)

# Plotting the examples
def plot_summary_method(title, original_text, summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    text = f"Original Text:\n{original_text[:500]}...\n\nSummary:\n{summary}"
    ax.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_summary_method("TF-IDF Summarization (CNN)", cnn_example, tfidf_summary_cnn)
plot_summary_method("TextRank Summarization (CNN)", cnn_example, textrank_summary_cnn)
plot_summary_method("T5 Summarization (CNN)", cnn_example, t5_summary_cnn)

plot_summary_method("TF-IDF Summarization (Amazon)", amazon_example, tfidf_summary_amazon)
plot_summary_method("TextRank Summarization (Amazon)", amazon_example, textrank_summary_amazon)
plot_summary_method("T5 Summarization (Amazon)", amazon_example, t5_summary_amazon)

# Evaluate summaries using ROUGE (only for CNN/Daily Mail dataset)
rouge = Rouge()
tfidf_scores = []
textrank_scores = []
t5_scores = []

for i in range(len(cnn_texts)):  # Evaluate only on CNN/Daily Mail dataset
    original_summary = cnn_summaries[i]
    tfidf_summary = tfidf_summarizer(cnn_texts[i])
    textrank_summary = textrank_summarizer(cnn_texts[i])
    t5_summary = t5_summarizer(cnn_texts[i])
    
    tfidf_score = rouge.get_scores(tfidf_summary, original_summary, avg=True)
    textrank_score = rouge.get_scores(textrank_summary, original_summary, avg=True)
    t5_score = rouge.get_scores(t5_summary, original_summary, avg=True)
    
    tfidf_scores.append(tfidf_score)
    textrank_scores.append(textrank_score)
    t5_scores.append(t5_score)

# Compute average ROUGE scores
avg_tfidf_scores = {key: np.mean([score[key]['f'] for score in tfidf_scores]) for key in tfidf_scores[0]}
avg_textrank_scores = {key: np.mean([score[key]['f'] for score in textrank_scores]) for key in textrank_scores[0]}
avg_t5_scores = {key: np.mean([score[key]['f'] for score in t5_scores]) for key in t5_scores[0]}

# Plot ROUGE scores
methods = ['TF-IDF', 'TextRank', 'T5']
rouge_1 = [avg_tfidf_scores['rouge-1'], avg_textrank_scores['rouge-1'], avg_t5_scores['rouge-1']]
rouge_2 = [avg_tfidf_scores['rouge-2'], avg_textrank_scores['rouge-2'], avg_t5_scores['rouge-2']]
rouge_l = [avg_tfidf_scores['rouge-l'], avg_textrank_scores['rouge-l'], avg_t5_scores['rouge-l']]

x = np.arange(len(methods))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, rouge_1, width, label='ROUGE-1')
rects2 = ax.bar(x, rouge_2, width, label='ROUGE-2')
rects3 = ax.bar(x + width, rouge_l, width, label='ROUGE-L')

ax.set_xlabel('Methods')
ax.set_ylabel('ROUGE Scores')
ax.set_title('ROUGE Scores by Summarization Method')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

fig.tight_layout()
plt.show()

# Print average ROUGE scores
print(f"Average ROUGE scores for TF-IDF: {avg_tfidf_scores}")
print(f"Average ROUGE scores for TextRank: {avg_textrank_scores}")
print(f"Average ROUGE scores for T5: {avg_t5_scores}")
