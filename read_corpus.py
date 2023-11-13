import os
import pickle
from typing import List

import pandas as pd
from tqdm import tqdm


class Token(object):
    def __init__(self, surface, lang, speaker, postag, length, ipa_len, bin_cognate, cognatehood, ner=''):
        self.surface = surface
        self.lang = lang
        self.speaker = speaker
        self.postag = postag
        self.len = length
        self.ipa_len = ipa_len
        self.bin_cognate = bin_cognate
        self.cognatehood = cognatehood
        self.ner = ner

    def __str__(self):
        s = "The token is: " + self.surface
        s += '\n'
        return s


class Utterance(object):
    def __init__(self, tokens, speaker, filename, uter_id):
        self.tokens = tokens
        self.speaker = speaker
        self.filename = filename
        self.uter_id = uter_id

    def __str__(self):
        s = ''
        for token in self.tokens:
            s += ' ' + str(token.surface)
        return s.strip()


def collect_all_utterances(dirname, filename):
    list_of_collected_utterances = []

    full_path_to_file = os.path.join(dirname, filename)
    df = pd.read_csv(full_path_to_file)

    utterances = df['uttId']

    unique_utterances = utterances.drop_duplicates()
    data_to_pick_up = ['speakerId', 'langId', 'word', 'posTag', 'word_length', 'length_in_phonemes_ipa',
                       'isCognate', 'cognatehood']

    for utterance_number in unique_utterances:
        selected_data = df[(df['uttId'] == utterance_number)]
        selected_data_to_choose_from = selected_data[data_to_pick_up].values.tolist()
        token_id = 0
        tokens = []
        for datum in selected_data_to_choose_from:
            speaker, lang, surface, postag, word_length, ipa_length, bin_cognate, cognatehood = datum
            new_token = Token(surface=surface, lang=lang,
                              speaker=speaker, postag=postag,
                              length=word_length, ipa_len=ipa_length, bin_cognate=bin_cognate,
                              cognatehood=cognatehood)
            tokens.append(new_token)
            token_id += 1

        list_of_collected_utterances.append(Utterance(tokens=tokens, speaker=speaker,
                                                      filename=filename, uter_id =utterance_number))

    return list_of_collected_utterances

def read_corpus(root_dir):
    filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    utterances = []
    for filename in tqdm(filenames):
        if not filename.endswith('.csv'):
            continue
        utterances.extend(collect_all_utterances(root_dir, filename))
    return utterances

def raw_csv_to_dat():
    utterances = read_corpus('bm_tagged_w_cognatehood_streched')
    with open("corpus/all_uter_with_cognatehood.dat", "wb") as f:
        pickle.dump(utterances, f)

    print(len(utterances))

    # Filter only CS utterances
    cs_uters = []
    for uter in utterances:
        lngs = set()
        for token in uter.tokens:
            if token.lang.startswith('shared'):
                continue
            lngs.add(token.lang)
        if len(lngs) > 1:
            cs_uters.append(uter)

    print(len(cs_uters))
    with open("corpus/cs_uters_with_cognatehood.dat", "wb") as f:
        pickle.dump(cs_uters, f)


if __name__ == "__main__":
    raw_csv_to_dat()
    pass