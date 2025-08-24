import nltk
import re
import heapq
from nltk.corpus import stopwords


nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

text = """Machine Learning (ML) is a crucial subset of Artificial Intelligence 
that focuses on building systems capable of learning from data. 
It enables applications like recommendation systems, 
fraud detection, self-driving cars, and medical diagnosis. 
ML algorithms improve automatically through experience, 
making predictions more accurate over time. 
However, challenges such as data privacy, algorithmic bias, 
and explainability remain important considerations in the deployment of ML technologies."""


text = re.sub(r'\s+', ' ', text.strip())


sentences = nltk.sent_tokenize(text)


stop_words = set(stopwords.words('english'))


word_freq = {}
for word in nltk.word_tokenize(text.lower()):
    if word.isalnum() and word not in stop_words:
        word_freq[word] = word_freq.get(word, 0) + 1


max_freq = max(word_freq.values())
for word in word_freq:
    word_freq[word] /= max_freq


sentence_score= {}
for sent in sentences:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_freq:
            sentence_score[sent] = sentence_score.get(sent, 0) + word_freq[word]

N = 2
summary_sent  = heapq.nlargest(N, sentence_score, key=sentence_score.get)
summary = ' '.join(summary_sent)


print("Original Text:\n", text)
print("\nSummary:\n", summary)
