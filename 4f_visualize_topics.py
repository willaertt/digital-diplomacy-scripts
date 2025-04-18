''' 
Visualize the inter-topic distances, annotate for topics analysed in the paper
'''

#import libraries
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertopic import BERTopic
import os
from umap import UMAP

if __name__ == "__main__":

    #specify paths to data files
    topic_model_files = "outputs/topic_models/"

    for topic_model_directory in os.listdir(topic_model_files):

        if not topic_model_directory.endswith('70_updated'): #only visualize the topic model used in the paper
            continue

        topic_model_path = os.path.join(topic_model_files, topic_model_directory)
        print('working on topic model ', topic_model_directory)

        print('load stored topic model')
        sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        topic_model = BERTopic.load(topic_model_path, embedding_model=sentence_model)
        topics = topic_model.topics_

        #define special topic categories
        multipolar = [0, 11, 95, 105, 245]
        great_patriotic_war = [3, 189, 153, 79]
        nazism = [17, 47, 263, 116, 181]
        golden_billion = [279]
        weapons_of_mass_destruction = [7, 32, 294, 282]

        category_colors = {
            'multipolar': ('rgba(0, 100, 100, 0.5)', 'rgba(0, 100, 100, 1)'),  # deep teal
            'great_patriotic_war': ('rgba(255, 0, 0, 0.8)', 'rgba(255, 0, 0, 1)'),  # red
            'nazism': ('rgba(0, 128, 0, 0.8)', 'rgba(0, 128, 0, 1)'),              # green
            'golden_billion': ('rgba(255, 165, 0, 0.8)', 'rgba(255, 165, 0, 1)'),  # orange
            'weapons_of_mass_destruction': ('rgba(128, 0, 128, 0.8)', 'rgba(128, 0, 128, 1)'),  # purple
        }

        topic_color_map = {}
        for tid in multipolar:
            topic_color_map[tid] = category_colors['multipolar']
        for tid in great_patriotic_war:
            topic_color_map[tid] = category_colors['great_patriotic_war']
        for tid in nazism:
            topic_color_map[tid] = category_colors['nazism']
        for tid in golden_billion:
            topic_color_map[tid] = category_colors['golden_billion']
        for tid in weapons_of_mass_destruction:
            topic_color_map[tid] = category_colors['weapons_of_mass_destruction']

        print('visualize custom intertopic distance map')
        for j in [3]:  # seed
            print(j)
            umap_model = UMAP(random_state=j)
            topic_model.umap_model = umap_model

            fig = topic_model.visualize_topics(top_n_topics=None)
            fig['layout'].pop('sliders', None)
            fig['layout'].pop('updatemenus', None)

            scatter = fig.data[0]
            x = scatter['x']
            y = scatter['y']
            customdata = scatter['customdata']

            topic_info = topic_model.get_topic_info()
            topics = topic_info[topic_info.Topic != -1]['Topic'].tolist()

            sample_step = 2
            sampled_topics = topics[::sample_step]

            special_topics = list(set(
                multipolar + great_patriotic_war + nazism + golden_billion + weapons_of_mass_destruction
            ))
            sampled_topics = list(set(sampled_topics + special_topics))

            annotations = []
            label_positions = {}
            marker_colors = []

            initial_offset = 0.15
            vertical_offset = 0.3
            max_attempts = 10
            min_distance = 0.15

            def is_too_close(new_x, new_y, label_positions, min_distance):
                for (existing_x, existing_y) in label_positions:
                    if np.sqrt((new_x - existing_x) ** 2 + (new_y - existing_y) ** 2) < min_distance:
                        return True
                return False

            for i in range(len(x)):
                topic_id = customdata[i][0]
                if topic_id == -1:
                    marker_colors.append('rgba(0, 0, 255, 0.1)')
                    continue

                node_color, label_color = topic_color_map.get(
                    topic_id,
                    ('rgba(0, 0, 255, 0.1)', 'rgba(0, 0, 0, 0.3)')
                )
                marker_colors.append(node_color)

                if topic_id not in sampled_topics:
                    continue

                topic_keywords = topic_model.get_topic(topic_id)
                if topic_keywords:
                    label = str(topic_id) + '. ' + ', '.join([word for word, _ in topic_keywords[:2]])
                else:
                    label = ""

                new_x = x[i]
                new_y = y[i]
                attempt = 0
                while is_too_close(new_x, new_y, label_positions, min_distance) and attempt < max_attempts:
                    new_x = x[i] + (attempt + 1) * initial_offset
                    new_y = y[i] + (attempt + 1) * vertical_offset
                    attempt += 1

                if topic_id == 0:
                    print(new_x, new_y)

                if attempt == max_attempts:
                    print(f"Max attempts reached for topic {topic_id}, label might still overlap")

                label_positions[(new_x, new_y)] = label

                font_size = 9 if topic_id in topic_color_map else 6

                annotations.append(
                    dict(
                        x=new_x,
                        y=new_y,
                        text=label,
                        xanchor='center',
                        yanchor='bottom',
                        showarrow=False,
                        font=dict(size=font_size, color=label_color)
                    )
                )

            #manually add annotation for Topic 0
            annotations.append(
                dict(
                    x=-2.4594183,
                    y=12.6, #13.025607
                    text="0. нато, украине",
                    xanchor='center',
                    yanchor='bottom',
                    showarrow=False,
                    font=dict(size=9, color='rgba(0, 100, 100, 1)'),  #use same color as multipolar
                )
            )

            fig.update_traces(
                marker=dict(
                    color=marker_colors,
                    line=dict(width=0),
                    size=scatter['marker']['size']
                )
            )

            fig.update_layout(
                annotations=annotations,
                title=None,
                sliders=[],
                updatemenus=[],
                showlegend=False,
                width=800,
                height=800,
                margin=dict(t=50, b=50, l=50, r=50)
            )

            print('save figure')
            fig.write_image('outputs/figures/fig8_intertopic_distance.pdf', scale=3)
