import re
from nltk import ngrams
from collections import Counter


def output_parser(paragraph):
    patterns = ["Thought:", "Action:", "Action Input:", "Observation:", "Final Answer:"]
    regex_pattern = '|'.join(map(re.escape, patterns))
    split_text = re.split(regex_pattern, paragraph)
    if split_text[0] == '':
        split_text.pop(0)
        if len(split_text) == 0:
            return []
         
    info_list = []
    if paragraph.startswith(split_text[0]):
        info_list.append(['', split_text[0]])
        paragraph = paragraph[len(split_text[0]):]
        split_text=split_text[1:]
    cur_text = 0
    while len(paragraph) > 0:
        key = paragraph.split(":")[0]
        if key + ":" not in patterns:
            print(paragraph)
            print(split_text)
            assert 1==0
        paragraph = paragraph[len(key) + 1:]
        value = split_text[cur_text]
        paragraph = paragraph[len(value):]
        cur_text += 1
        info_list.append([key,value.strip()])
    return info_list


def truncate_at_marker(text, marker="[/INST]"):

    if isinstance(text, dict):
        text = "\n".join([f"{key}: {value}" for key, value in text.items()])
    index = text.find(marker)
    if index != -1:
        return text[:index]
    return text


def detect_overly_long_sequences(text):
    long_sequence_pattern = r"\b\w{50,}\b"  # words longer than 40 characters
    long_words = re.findall(long_sequence_pattern, text)
    for word in long_words:
        if "http" not in word:
            return True


def detect_repetitive_language(text, n_gram = 8, n_rep = 10):
    words = text.split()
    n_grams = list(ngrams(words, n_gram))
    frequency = Counter(n_grams)
    if len(frequency) == 0:
        return False
    if max(frequency.values()) >= n_rep:
        return True
    return False
