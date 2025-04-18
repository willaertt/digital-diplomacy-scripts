'''
Load the evaluation dataframe, format it to create table for the methodological appendix
'''

#import libraries
import pandas as pd

#load evaluation scores
evaluation_df = pd.read_csv('outputs/lists/topic_model_evaluation.csv')
evaluation_df

#add columns with min docs and topic count
min_docs = [90, 70, 50, 100, 60, 80]
topic_count = [311, 414, 651, 258, 517, 345]
evaluation_df['min_docs_per_topic'] = min_docs
evaluation_df['topic_count'] = topic_count

#sort dataframe by min docs, clean columns
evaluation_df = evaluation_df.sort_values(by='min_docs_per_topic')
evaluation_df = evaluation_df[['min_docs_per_topic', 'topic_count', 'u_mass', 'c_v', 'c_uci', 'c_npmi', 'proportion_unique_words', 'pairwise_jaccard_diversity', 'irbo']]

#round values to three decimals
evaluation_df = evaluation_df.round(3)
evaluation_df

#store the updated dataframe
evaluation_df.to_csv('outputs/lists/topic_model_evaluation_clean.csv', index = False)
