'''
We loop over each of the topic models we have created, and finetune their representation
Store information about the topics (their hierarchies etc.)
We then annotate each message in the dataset for the retrieved topics (for the different sizes)
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

        print('save topic information')
        topic_info_df = topic_model.get_topic_info()
        topic_info_df = topic_info_df.rename(columns={'Topic': 'topic_number', 'Count': 'topic_count', 'Name':'topic_name', 'CustomName': 'topic_custom_name', 'Representation': 'topic_representation', 'Representative_Docs': 'topic_representative_docs'}) 
        topic_info_df.to_csv(os.path.join(updated_topic_model_directory, 'topic_model_info.csv'), index=False)

        print('save document-level information of topics')
        document_info = topic_model.get_document_info(docs)
        document_info.to_csv(os.path.join(updated_topic_model_directory, 'document_info.csv'), index=False)

        #save visualizations for the topics
        print('save topic visualization')
        fig = topic_model.visualize_topics(custom_labels= True)
        fig.write_html(os.path.join(updated_topic_model_directory, "topic_visualization.html"))

        #save hierarchical topic representation
        print('save topic hierarchical representation as html')
        fig_hierarchy = topic_model.visualize_hierarchy(custom_labels = True)
        fig_hierarchy.write_html(os.path.join(updated_topic_model_directory, "topic_hierarchy.html"))

        print('finetune topic hierarchy figure and save as png')
        fig_hierarchy = topic_model.visualize_hierarchy(custom_labels = True)
        fig_hierarchy.update_layout(width=1300, height= 8000, title_text='')
        fig_hierarchy.write_image(os.path.join(updated_topic_model_directory, 'topic_hierarchy.png'), scale =3)

        #save bar chart for top n topics
        print('save bar chart for top n topics')
        n = 10
        fig_barchart = topic_model.visualize_barchart(top_n_topics=n, custom_labels= True)
        fig_barchart.write_html(os.path.join(updated_topic_model_directory, "topic_barchart.html"))

        print('save dataset with annotations for topics')
        topic_size_string = topic_model_path.split('_')[-1] #get the topic model min cluster size string to name the topic column
        filtered_df['topic_number'] = topic_model.topics_ 
        annotated_embassy_df = embassy_df.merge(filtered_df[['original_index', 'topic_number']], on='original_index', how='left') 
        annotated_embassy_df = annotated_embassy_df.merge(topic_info_df[['topic_number', 'topic_count', 'topic_name', 'topic_custom_name', 'topic_representation', 'topic_representative_docs']], on = 'topic_number', how = 'left')
        annotated_embassy_df.to_csv(os.path.join(updated_topic_model_directory, 'sample_with_topic_annotations.csv'), index = False)