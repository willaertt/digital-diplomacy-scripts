'''
This script is used to analyse the linguistic diversity of the dataset (to check its coverage by the multilingual BERTopic model)
List of languages covered by the multilingual model: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/blob/main/README.md

We find that the following languages are not (explicitly) supported by this model: ['tl', 'so', 'ta', 'no', 'ne', 'af', 'bn', 'cy', 'sw']
In total, we find 324 messages that contain one of these languages, which are thus not supported by the model 
'''

#import libraries
import pandas as pd
from ast import literal_eval

#set path to data
csv_sample_file = "/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data_sample.csv"
 
#specify list of languages based on model documentation (https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/blob/main/README.md)
model_languages= ['ar', 'bg','ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'fi', 'fr', 'gl', 'gu', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'it', 'ja', 'ka', 'ko', 'ku', 'lt', 'lv', 'mk', 'mn', 'mr', 'ms', 'my', 'nb', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sq', 'sr', 'sv', 'th', 'tr', 'uk', 'ur', 'vi']


#load the data
print('load data')
embassy_df = pd.read_csv(csv_sample_file)
languages_df = embassy_df[embassy_df['message_language'].notnull()]
languages_df['message_language'] = languages_df['message_language'].apply(literal_eval)

#get list of languages
languages = list(languages_df['message_language'])
languages = list(set([x for xs in languages for x in xs])) #flatten list

#identify languages in the data that are not part of the model
languages_not_in_model = list(set(languages) - set(model_languages))
print('languages not in model:', languages_not_in_model)

#total number of messages that include these languages
counter = 0
for list in list(languages_df['message_language']):
    if set(list) & set(languages_not_in_model):
        counter +=1 

print('number of messages with language not in model', counter)
print('number of messages with clean text that is not empty', len(embassy_df[embassy_df['clean_message'].notnull()].index))
print('number of messages with language detection that is not empty', len(languages_df.index))