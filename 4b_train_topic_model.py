'''
Pipeline for training and storing topic models with different minimum topic sizes

Topic modelling using BERTopic
https://maartengr.github.io/BERTopic/index.html#quick-start 

@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}


model choice: https://maartengr.github.io/BERTopic/faq.html#why-are-the-results-not-consistent-between-runs
model documentation: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/blob/main/README.md 

'''

#import libraries
import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":

    #specify paths to data files
    csv_sample_file = "/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data_sample.csv"
   
    #load data
    print('load data')
    embassy_df = pd.read_csv(csv_sample_file)
    print('sample size', len(embassy_df.index))

    #filter messages where text is not empty, get docs
    filtered_df = embassy_df[embassy_df['clean_message'].notna()]
    docs = filtered_df['clean_message'].tolist()

    #train the topic model with different minimum topic sizes
    min_topic_sizes = [50, 60, 70, 80, 90, 100]

    for topic_size in min_topic_sizes:

      print('train topic model with min size' , str(topic_size))

      #check if the directory for storing the topic model exists, otherwise create it
      topic_model_folder = "/home/tom/Documents/code/GitHub/geopolitics-propaganda-new/outputs/topic_models/" + "embassy_topic_model_" + str(topic_size)
      if not os.path.isdir(topic_model_folder):
        print('create topic model folder')
        os.mkdir(topic_model_folder)

      #load the BERTopic multilingual model
      embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
      topic_model = BERTopic(embedding_model=embedding_model, min_topic_size= topic_size) #use large multilingual model, train for multiple different sizes

      #fit the model to the messages
      print('fit topic model')
      topics, probabilities = topic_model.fit_transform(docs)

      #store the topic model
      print('save topic model to disk')
      topic_model.save(topic_model_folder, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)