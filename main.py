from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords,state_union
import re
import math
from collections import defaultdict, Counter
import heapq
import tkinter as tk
from tkinter import ttk


def preprocess(sentence):
    s = ""
    stop_w = stopwords.words("english")
    for i in word_tokenize(sentence):
        if i not in stop_w:
            s = s + i + " "
    se = re.sub(r'[^\w\s]', '', s)
    se = se.lower()
    return se


class Node:
    def __init__(self, id: int, text: str):
        self.id = id
        self.text = text


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.edges = {}
        for i in range(len(self.nodes)):
            self.edges[i] = []
        self.d = 0.85
        self.max_iter = 100

    def add_edge(self, u: int, v: int):
        self.edges[u].append(v)

    def outgoing(self, k: int) -> int:
        return len(self.edges[k])

    def incoming(self, j: int):
        l = []
        for i in range(len(self.nodes)):
            if j in self.edges[i]:
                l.append(i)
        return l

    def process(self, heap, l, j):
        s = 0
        inc = self.incoming(j)
        for i in inc:
            count = self.outgoing(i)
            if count > 0:
                s += l[i] / count
        return s

    def pagerank(self):
        scores = [1 / len(self.nodes) for i in range(len(self.nodes))]
        min_heap = []
        for i in range(len(self.nodes)):
            heapq.heappush(min_heap, (-scores[i], i))
        for i in range(self.max_iter):
            new_scores = [0 for i in range(len(self.nodes))]
            for j in range(len(self.nodes)):
                node = self.nodes[j]
                x = scores[j]
                new_score = (1 - self.d) * x + self.d * self.process(min_heap, scores, j)
                new_scores[j] = new_score
                heapq.heappush(min_heap, (-new_score, j))
            scores = new_scores
        return scores

    def summarize(self, n):
        scores = self.pagerank()
        ranked_sentences = [(score, self.nodes[i].text) for i, score in enumerate(scores)]
        ranked_sentences.sort(reverse=True)
        num_sentences = n
        summary = [sentence for score, sentence in ranked_sentences[:num_sentences]]
        return summary


class PageRankSummarizer:
    def __init__(self, n):
        self.n = n
        self.summary = ""

    def compute_similarity(self, text1, text2):
        def get_word_counts(text):
            words = re.findall(r'\w+', text.lower())
            return Counter(words)
        
        word_counts1 = get_word_counts(text1)
        word_counts2 = get_word_counts(text2)
        words = set(word_counts1.keys()) | set(word_counts2.keys())
        dot_product = sum(word_counts1.get(word, 0) * word_counts2.get(word, 0) for word in words)
        magnitude1 = math.sqrt(sum(count ** 2 for count in word_counts1.values()))
        magnitude2 = math.sqrt(sum(count ** 2 for count in word_counts2.values()))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def pagerank_summarize(self, text):
        nodes = [Node(i, sentence) for i, sentence in enumerate(sent_tokenize(text))]
        graph = Graph(nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity = self.compute_similarity(nodes[i].text, nodes[j].text)
                if similarity > 0.5:
                    graph.add_edge(i, j)
                    graph.add_edge(j, i)

        summary_sentences = graph.summarize(self.n)
        summary = " ".join(summary_sentences)
        return summary



class TFIDFSummarizer:
    def __init__(self, n):
        self.n = n
        self.tfidf_scores = defaultdict(dict)
        self.sentence_scores = []

    def fit(self, sentences):
        for i, sentence in enumerate(sentences):
            tf_scores = self._calculate_tf(sentence)
            idf_scores = self._calculate_idf(sentence)
            for word in tf_scores:
                if word in idf_scores:
                    self.tfidf_scores[word][i] = tf_scores[word] * idf_scores[word]

    def _calculate_idf(self, sentences):
        idf_s = defaultdict(int)
        num_lines = len(sentences)
        for sentence in sentences:
            words = set(sentence.split())
            for wo in words:
                idf_s[wo] += 1
        for wo in idf_s:
            idf_s[wo] = math.log(num_lines / idf_s[wo])
        return idf_s

    def _calculate_tf(self, sentence):
        tf_scores = defaultdict(int)
        words = sentence.split()
        word_count = len(words)
        for w in words:
            tf_scores[w] += 1 / word_count
        return tf_scores

    def summarize(self, sentences):
        for i, sentence in enumerate(sentences):
            sentence_score = sum(self.tfidf_scores[word].get(i, 0) for word in sentence.split())
            heapq.heappush(self.sentence_scores, (sentence_score, sentence))
            if len(self.sentence_scores) > self.n:
                heapq.heappop(self.sentence_scores)
        self.summary = [sentence for _, sentence in
                        sorted(self.sentence_scores, reverse=True)] 
    def printSummary(self):
        j = 0
        sum = ""
        for i in self.summary:
            print(i)
            sum = sum+i
            if j >= self.n:
                break
            j = j + 1
        return sum



def button_clicked():
    text1 = entry1.get("1.0", tk.END)
    dropdown_value = dropdown.get()
    number_value = number.get()

    if dropdown_value == 'TF-IDF':
        example_text = sent_tokenize(text1)
        preprocessed_sentences = [preprocess(sentence) for sentence in example_text]
        summarizer = TFIDFSummarizer(int(number_value))
        summarizer.fit(preprocessed_sentences)
        summarizer.summarize(example_text)
        summary = summarizer.printSummary()

    elif dropdown_value == "PageRank":
        pr_summarizer = PageRankSummarizer(int(number_value))
        summary = pr_summarizer.pagerank_summarize(text1)

    entry2.delete('1.0', tk.END)
    entry2.insert('1.0', summary)


root = tk.Tk()
root.title("Text Summarizer")

label1 = tk.Label(root, text="Text Field 1:")
label1.grid(row=0, column=0, sticky=tk.W)
entry1 = tk.Text(root, width=65, height=40)
entry1.grid(row=1, column=0, padx=10)

label3 = tk.Label(root, text="Dropdown:")
label3.grid(row=1, column=1)
dropdown = ttk.Combobox(root, values=["TF-IDF", "PageRank"])
dropdown.grid(row=1, column=2)

label4 = tk.Label(root, text="Number:")
label4.grid(row=1, column=3)
number = tk.Entry(root)
number.grid(row=1, column=4)

label2 = tk.Label(root, text="Text Field 2:")
label2.grid(row=0, column=5, sticky=tk.W)
entry2 = tk.Text(root, width=65, height=40)
entry2.grid(row=1, column=5, padx=10)

button = tk.Button(root, text="Summarize!", command=button_clicked)
button.grid(row=2, column=2, pady=10)

root.mainloop()
