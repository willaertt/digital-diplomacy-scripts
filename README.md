# digital-diplomacy-scripts

 This repository contains python scripts accompanying the paper "From Denazification to The Golden Billion: An Inductive Analysis of the  Kremlin's Weaponization of Digital Diplomacy on Telegram"

 The Telegram dataset on which the paper is based can be retrieved from https://zenodo.org/records/13494674

## overview of scripts

The repository contains the following scripts that support the analysis reported on in the paper:

1. 1_preprocessing.py: This script is used to clean and preprocess the raw telethon data. It is used to remove urls, hashtags, mentions, detect language(s), and identify sources of forwarded messages. 

2. 2_channel_composition_and_activity.py: This script is used in support of the descriptive analysis of the dataset, focusing on linguistic composition and channel activity. 

3. 3_network_analysis.py: This script plots the information flow between channels based on forwarded messages. A full analysis of the forwarding network and functions for sampling the graph are included.  

4. 4a_check_model_languages.py: This script is used to analyse the linguistic diversity of the dataset and to check its coverage by the multilingual embedding model.

5. 4b_train_topic_model.py: This script contains the ipeline for training and storing topic models with different minimum topic sizes. 

6. 4c_finetune_topic_model.py: This script is used for finetuning representations of topics and for storing information about the topics. 

7. 4d_topic_model_evaluation.py: Script for evaluating topic models. Returns scores for topic coherence and diversity.

8. 4e_analyse_evaluation.py: Script for creating tabular overview of topic model evaluations. 

9. 4f_visualize_topics.py: Script for visualizing the inter-topic distances of BERTopic model, annotated for the topics discussed in the paper.
