import stanza
from tqdm import tqdm
import os
import pandas as pd


def parse_cognate_file(lng):
    """
    :param lng: en / es
    :return: dict of Yuli's file
    """
    res = {}
    with open(f'{lng}_cognates_final.txt') as f:
        lines = f.readlines()
        for l in lines:
            split = l.strip().split()
            res[split[0]] = split[3]

    return res


def tag_corpus(root_dir):
    filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    for filename in tqdm(filenames):
        if not filename.endswith('.csv'):
            continue

        tag_all_utterances(root_dir, filename)


def lemma_score(word, nlp, cognate_dict):
    lemma = nlp(word).sentences[0].words[0].lemma
    if not lemma:
        return 0
    else:
        lemma = lemma.lower()
        cognatehood_score = cognate_dict.get(lemma, 0)
        return cognatehood_score


def tag_all_utterances(dirname, filename):
    full_path_to_file = os.path.join(dirname, filename)
    df = pd.read_csv(full_path_to_file)

    cognatehood = []

    utterances = df['uttId']

    unique_utterances = utterances.drop_duplicates()
    data_to_pick_up = ['word', 'langId']

    for utterance_number in unique_utterances:
        selected_data = df[(df['uttId'] == utterance_number)]
        selected_data_to_choose_from = selected_data[data_to_pick_up].values.tolist()

        words = [w[0] for w in selected_data_to_choose_from]
        langs = [w[1] for w in selected_data_to_choose_from]

        for word, lang in zip(words, langs):
            if lang == 'eng':
                cognatehood.append(lemma_score(word, english_nlp, english_cognate_dict))
            elif lang == 'spa':
                cognatehood.append(lemma_score(word, spanish_nlp, spanish_cognate_dict))
            else:
                cognatehood.append(0)



    df['cognatehood'] = cognatehood
    new_path = os.path.join('../bm_tagged_w_cognatehood', filename)
    df.to_csv(new_path)



english_cognate_dict = parse_cognate_file('en')
spanish_cognate_dict = parse_cognate_file('es')


english_nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma')
spanish_nlp = stanza.Pipeline(lang='es', processors='tokenize,lemma')

tag_corpus('../bm_tagged')
