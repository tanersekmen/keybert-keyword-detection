from constant import stop_words
import re


def remove_stopwords(text):
    words = re.findall(r'\b\w+\b', text)
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text
