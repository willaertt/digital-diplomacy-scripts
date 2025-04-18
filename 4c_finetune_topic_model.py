'''
We loop over each of the topic models we have created, and finetune their representation
We store the updated topic models
'''

#import libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertopic import BERTopic
import pandas as pd 
import os


if __name__ == "__main__":

    #specify paths to data files
    csv_sample_file = "/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data_sample.csv"
    topic_model_files = "outputs/topic_models/"

    #get data and docs
    #load data
    print('load data')
    embassy_df = pd.read_csv(csv_sample_file)
    print('sample size', len(embassy_df.index))

    #get docs
    print('get docs')
    filtered_df = embassy_df[embassy_df['clean_message'].notna()]
    docs = filtered_df['clean_message'].tolist()
    print('number of docs', len(docs))

    #load the topic models of different sizes
    for topic_model_directory in os.listdir(topic_model_files):

        #do not run the updates on any other topic model than the one with min_cluster_size 70 (comment out to run on all models)
        if not topic_model_directory.endswith('70_updated'):
            continue

        #do not run if model has already been updated
        if topic_model_directory.endswith('_updated'):
            continue

        #set path to topic model
        topic_model_path = os.path.join(topic_model_files, topic_model_directory)
        print('working on topic model ', topic_model_directory)

        #load topic model from disk with the sentence model that was used to train it
        print('load stored topic model')
        sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        topic_model = BERTopic.load(topic_model_path, embedding_model = sentence_model)
        old_topics = topic_model.topics_

        #update topic representations
        print('reduce outliers, update the topic representations')
        new_topics = topic_model.reduce_outliers(docs, old_topics)
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), max_df = 0.85)
        topic_model.update_topics(docs, vectorizer_model=vectorizer_model, topics = new_topics)

        #set parameters for topic models
        topic_labels = topic_model.generate_topic_labels(nr_words=5,
                                                        topic_prefix=True,
                                                        word_length=20,
                                                        separator=", ")

        print('update topic model labels')
        topic_model.set_topic_labels(topic_labels)

        #save the updated topic model and topic model information
        print('save updated topic model and topic model information')
        updated_topic_model_directory = os.path.join(topic_model_files, topic_model_directory + '_updated')
        if not os.path.isdir(updated_topic_model_directory):
            print('create topic model folder:', updated_topic_model_directory)
            os.mkdir(updated_topic_model_directory)

        print('save updated topic model')
        embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        topic_model.save(updated_topic_model_directory, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)