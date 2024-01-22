from keybert import KeyBERT
import pandas as pd
from collections import Counter
from constant import path, model_name, n_keywords, ngram
import matplotlib.pyplot as plt
from utils import remove_stopwords


def get_data(path, col, encoding='utf-8'):
    df = pd.read_csv(path, encoding = encoding)
    def remove_quotes(text):
        if text.startswith("'") and text.endswith("'"):
            return text[1:-1]
        else:
            return text
    df[col] = df[col].apply(remove_quotes)
    return df 


def set_sentence(sentence:str):
    result = sentence.lower().replace('ö', 'o').replace('ı', 'i').replace('ü', 'u').replace('ç', 'c').replace('ğ', 'g').replace('ş', 's')
    return result


def check_drop_duplicate(data, col):
    data = data[data.duplicated(subset = col, keep = False)]
    data = data.reset_index()
    data = data[[col]]
    return data


def prepare_model(data, model_name, n_keywords, 
                  ngram, use_mmr = True, 
                  highlight = False, diversity = 0.5):
    kw_model = KeyBERT(model=model_name)
    key_det = data.apply(lambda x: kw_model.extract_keywords(x, 
                                                             keyphrase_ngram_range=(1, ngram),
                                                             top_n=n_keywords,
                                                             diversity=diversity,
                                                             use_mmr=use_mmr,
                                                             highlight=highlight))
    new_df = key_det['text'].tolist()

    return new_df


def show_graph(new_df):
    all_keywords = [item[0] for sublist in new_df for item in sublist]
    keyword_counts = Counter(all_keywords)
    top_30_keywords = dict(keyword_counts.most_common(30))

    plt.figure(figsize=(18, 9))
    plt.bar(top_30_keywords.keys(), top_30_keywords.values())
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.title('Top 30 Keywords Frequency Bar Chart')
    plt.xticks(rotation='vertical')  
    plt.savefig('most_common_keywords.png')
    plt.show()
    return top_30_keywords
    
    
if __name__=='__main__':
    df = get_data(path, 'text')
    df['text'] = df.text.apply(set_sentence)
    df['text'] = df['text'].apply(remove_stopwords)
    print(df.head())
    df = check_drop_duplicate(df, 'text')
    key_det = prepare_model(df, model_name=model_name, 
                            n_keywords=n_keywords, ngram=ngram)
    common_keywords = show_graph(key_det)
    print(common_keywords)

    
   

