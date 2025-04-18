'''
A script for evaluating the quality of the topic models and reporting metrics for different values for minimum documents per topic
Uses diversity metrics from the repository https://github.com/silviatti/topic-model-diversity/blob/master/
'''

#import libraries
from bertopic import BERTopic
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertopic import BERTopic
import pandas as pd 
import os

import sys
sys.path.insert(1, '/home/tom/Documents/code/GitHub/topic-model-diversity')
from diversity_metrics import proportion_unique_words, pairwise_jaccard_diversity, irbo

#suppress hugging face warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#define function for calculating coherence score
def evaluate_topic_model(model_name, topics, docs, topk):
    '''
    Get coherence scores for BERTopic topic model, source: https://github.com/MaartenGr/BERTopic/issues/90
    Get diversity scores for BERTopic topic model, source: https://github.com/silviatti/topic-model-diversity/blob/master/
    '''
    
    #preprocess Documents
    print('preprocess documents')
    documents = pd.DataFrame({"Document": docs,
                            "ID": range(len(docs)),
                            "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    #extract vectorizer and analyzer from BERTopic
    print('extract vectorizer')
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    #extract features for Topic Coherence evaluation
    print('extract features for coherence evaluation')
    words = vectorizer.get_feature_names_out()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    #exclude outliers
    print('exclude outlier topic')
    topic_words = [
        [word for word, _ in topic_model.get_topic(topic)] #top 1° 
        for topic in documents_per_topic.Topic
        if topic != -1
    ]

    #evaluate coherence
    print('evaluate coherence')
    metrics = ["u_mass", "c_v", "c_uci", "c_npmi"]
    results = {"model": model_name}

    for metric in metrics:
        print('calculate ', metric)
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokens,
            corpus=corpus,
            dictionary=dictionary,
            coherence=metric
        )
        results[metric] = cm.get_coherence()

    #evaluate diversity
    print('evaluate diversity')
    results['proportion_unique_words'] = proportion_unique_words(topic_words, topk)
    results['pairwise_jaccard_diversity'] = pairwise_jaccard_diversity(topic_words, topk)
    results['irbo'] = irbo(topic_words, topk= topk, weight=0.5)

    return results



if __name__ == "__main__":

    #specify paths to data files
    csv_sample_file = "/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data_sample.csv"
    topic_model_files = "outputs/topic_models/"

    #get data and docs
    print('load data')
    embassy_df = pd.read_csv(csv_sample_file)
    print('sample size', len(embassy_df.index))

    #get docs
    print('get docs')
    filtered_df = embassy_df[embassy_df['clean_message'].notna()]
    docs = filtered_df['clean_message'].tolist()
    print('number of docs', len(docs))

    #initiate dataframe for model metrics
    result_dicts = []

    #load the topic models of different sizes, run the evaluations
    for topic_model_directory in os.listdir(topic_model_files):

        #only look at the updated topic models
        if not topic_model_directory.endswith('_updated'): #only get metrics for the update topic models
            continue

        #set path to topic model
        topic_model_path = os.path.join(topic_model_files, topic_model_directory)
        print('working on topic model ', topic_model_directory)

        #load topic model from disk with the sentence model that was used to train it
        print('load stored topic model')
        sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        topic_model = BERTopic.load(topic_model_path, embedding_model = sentence_model)
        topics = topic_model.topics_

        #chek whether the vectorizer model is correct
        print(topic_model.vectorizer_model.ngram_range)

        #analyse coherence
        print('evaluate topic model')
        results = evaluate_topic_model(topic_model_directory, topics, docs, topk = 10)
        print(results)
        result_dicts.append(results)

    #compile evaluation metrics into a single dataframe
    print('save metrics df')
    metrics_df = pd.DataFrame(result_dicts)
    metrics_df.to_csv('outputs/lists/topic_model_evaluation.csv', index = False)