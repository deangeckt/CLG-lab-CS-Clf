from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import spacy

en_nlp = spacy.load("en_core_web_md")
es_nlp = spacy.load("es_core_news_md")


def tag_corpus(root_dir):
    filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    for filename in tqdm(filenames):
        if not filename.endswith('.csv'):
            continue

        tag_all_utterances(root_dir, filename)

def tag_all_utterances(dirname, filename):
    full_path_to_file = os.path.join(dirname, filename)
    df = pd.read_csv(full_path_to_file)

    ner = []

    utterances = df['uttId']

    unique_utterances = utterances.drop_duplicates()
    data_to_pick_up = ['word', 'langId']

    for utterance_number in unique_utterances:
        selected_data = df[(df['uttId'] == utterance_number)]
        selected_data_to_choose_from = selected_data[data_to_pick_up].values.tolist()

        words = [w[0] for w in selected_data_to_choose_from]
        langs = [w[1] for w in selected_data_to_choose_from]
        unique, counts = np.unique(langs, return_counts=True)
        lang = unique[np.argmax(counts)]
        lang = lang if lang == 'spa' else 'eng'

        nlp = en_nlp if lang == 'eng' else es_nlp
        doc = list(nlp.pipe(words))
        for i, text in enumerate(doc):
            entity = text.ents[0].label_ if len(text.ents) else 'none'
            ner.append(entity)

    df['ner'] = ner
    new_path = os.path.join('bm_tagged_w_ner', filename)
    df.to_csv(new_path)





tag_corpus('bm_tagged')
