'''
Data preprocessing: get sample, remove urls, hashtags, mentions, detect language(s), identify sources of forwarded messages
'''

#import libraries
import os
import jsonlines
import pandas as pd
from tqdm import tqdm
from langdetect import detect_langs, lang_detect_exception
import re
from ast import literal_eval


#define helper functions
def get_forwarded_msg_df(embassy_df):
    '''
    returns a dataframe containing only forwarded messages
    '''

    forwarded_msg_df = embassy_df[~embassy_df['fwd_from'].isnull()]

    return forwarded_msg_df

def ndjson_to_csv(json_file, csv_file):
    '''
    Convert ndjson file with Telegram data to a csv file
    json_file = path to ndjson
    csv_file = path to csv
    '''
    
    #initiate list for raw data
    data = []

    #open the NDJSON file and read the data
    print('read ndjson')
    with jsonlines.open(json_file) as reader:
        for obj in tqdm(reader):
            data.append(obj)

    #convert the list of dictionaries to a pandas DataFrame
    print('convert to dataframe')
    df = pd.DataFrame(data)

    #write the DataFrame to a CSV file
    print('write dataframe to csv')
    df.to_csv(csv_file, index=False)

def create_sample(csv_file, csv_sample_file, sample_size):
    '''
    creates a pandas dataframe with a random sample of Telegram data
    json_file = path to raw data (ndjson)
    csv_file = path to data csv

    if start_from_raw_data == True: start by converting raw data to csv, load csv as dataframe, else start from csv
    sample_size = size of random sample
    sample_path = path to store sample 
    '''

    #load data
    if os.path.exists(csv_file):
        embassy_df = pd.read_csv(csv_file, low_memory = False)
    else:
        print('csv file does not yet exist, please start from raw data')

    print('get sample')
    embassy_df = embassy_df.sample(n=sample_size, random_state=1) #keep random state to ensure reproducibility of rsults
    
    print('store sample')
    embassy_df.to_csv(csv_sample_file, index = False)

    return embassy_df

#cleaning
def remove_emojis(text):
    ''' 
    remove emojis from 
    '''
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F" 
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF" 
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF" 
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)

#cleaning
def clean_string(string):
    '''
    clean string: remove URLs, mentions, hashtags and emojis
    '''

    #remove URLs
    string = re.sub(r'(https?:\/\/(?:\S+?\.)+\S+)\b', '', string)

    #remove mentions (e.g., @username)
    string = re.sub(r'@[a-zA-Z0-9(_)]+', ' ', string)
    
    #remove hashtags
    string = re.sub(r'#[a-zA-Z0-9(_)]+', ' ', string)

    #remove emojis
    string = remove_emojis(string)
        
    return string.strip()


#define language detection function
def detect_languages(string):
    '''
    Detect any of the languages in the text
    Uses langdetect library
    '''

    if type(string) is str:
        try:
            languages = detect_langs(string)
        except lang_detect_exception.LangDetectException:
            languages = None

    else: 
        languages = None

    return languages

def identify_message_languages(message):
    '''
    Take a Telegram message, split it by newlines, clean each line and identify its language
    returns a list of languages for each message
    '''

    #initiate list for appending identified languages
    message_languages = []

    #if there is no string, return None (no languages detected)
    if type(message) is not str:
        return None
        
    #split into lines
    message_lines = message.split("\n")

    for line in message_lines:

        #clean line
        clean_line = clean_string(line)

        #only look at strings longer than 5 words (e.g. to avoid things like "New York, 24 aprile 2023" to be detected as English)
        if len(clean_line.split(' ')) > 5: 
            
            #detect the languages in this string 
            languages = detect_languages(clean_line) 

            if not languages: #if the language detection returns 'None', continue
                continue

            #sort languages by probablity
            languages_sorted = sorted(languages, key=lambda x: x.prob, reverse=True)

            #select the most probable language
            most_probable_language = languages_sorted[0]

            #if the probability is less than 0.8, skip this line
            if most_probable_language.prob < 0.8:
                continue

            #append to list
            message_languages.append(str(most_probable_language.lang))

    if message_languages == []:
        return None


    return list(set(message_languages))


def return_fwd_source(dict):
    '''
    we try to identify the source channel (i.e. the channel from which the message is forwarded)
    if one of the keys does not exist or the 'fwd_from' field is None, we add an 'other' node to the network (these typically denote bots etc. that are not public channels)
    '''

    try:
        forwarded_channel_username = dict['from_id']['chats'][0]['username'] #Introduce a check here for forwarded messages where "from_id" is `None'
        if forwarded_channel_username is None:
            forwarded_channel_username = 'other' 

    except (TypeError, KeyError, IndexError): 
        forwarded_channel_username = 'other'

    return forwarded_channel_username


if __name__ == "__main__":

    #specify paths to data files
    raw_data_file = "/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data.ndjson"
    raw_csv_data_file = "/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data.csv"
    csv_sample_file = "/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data_sample.csv"

    #convert raw data to csv (optional)
    #print('convert data')
    #ndjson_to_csv(raw_data_file, raw_csv_data_file)

    #create random sample (optional)
    # print('create sample')
    # embassy_df = create_sample(raw_csv_data_file, csv_sample_file, sample_size = 400000)

    # #load data (optional)
    # print('load sample')
    # embassy_df = pd.read_csv(csv_sample_file)
    # print('sample size:', len(embassy_df.index))

    #start from full dataset
    print('load data')
    embassy_df = pd.read_csv(raw_csv_data_file) #choose raw data to start from scratch
    print('sample size:', len(embassy_df.index))

    #convert date column to datetime and filter the dataset for the daterange discussed in the paper
    print('filter daterange')
    start_date = '2020-02-01'
    end_date = '2024-03-01'
    embassy_df['date'] = pd.to_datetime(embassy_df['date'], unit = "s") 
    embassy_df = embassy_df[(embassy_df['date'] >= start_date) & (embassy_df['date'] < end_date)]
    print('sample size:', len(embassy_df.index))

    #get messages with text
    embassy_df['original_index'] = embassy_df.index
    filtered_df = embassy_df[embassy_df['message'].notna()]

    #clean text
    print('clean messages')
    filtered_df['clean_message'] = filtered_df['message'].apply(clean_string)

    #detect languages
    print('detect languages')
    filtered_df['message_language'] = filtered_df['clean_message'].apply(identify_message_languages)

    #annotate original data with clean text, languages
    print('annotate data with linguistic information')
    embassy_df = embassy_df.merge(filtered_df[['original_index', 'clean_message']], on='original_index', how='left')
    embassy_df = embassy_df.merge(filtered_df[['original_index', 'message_language']], on='original_index', how='left')

    #add column with source of forwarded message 
    print('get forwarding sources')
    forwarded_df = get_forwarded_msg_df(embassy_df)
    forwarded_df['fwd_from'] = forwarded_df['fwd_from'].apply(literal_eval)
    forwarded_df['fwd_source'] = forwarded_df['fwd_from'].apply(return_fwd_source)

    print('annotate data with forwarding information')
    embassy_df = embassy_df.merge(forwarded_df[['original_index', 'fwd_source']], on='original_index', how='left')

    #store data with additional columns
    print('save annotated sample')
    embassy_df.to_csv(csv_sample_file, index = False)