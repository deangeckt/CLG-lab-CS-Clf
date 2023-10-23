# version: 1.1
import os
import codecs
import pandas as pd

from collections import Counter

from classes import *
from language_analysis import *


def collect_all_utterances(dirname, filename):
    list_of_collected_utterances = []

    full_path_to_file = os.path.join(dirname, filename)
    # df = pd.read_csv(full_path_to_file, sep=',')
    df = pd.read_csv(full_path_to_file)

    utterances = df['uttId']

    unique_utterances = utterances.drop_duplicates()
    # data_to_pick_up = ['speaker', 'langid', 'surface', 'auto']
    data_to_pick_up = ['speakerId', 'langId', 'word']

    for utterance_number in unique_utterances:
        selected_data = df[(df['uttId'] == utterance_number)]
        selected_data_to_choose_from = selected_data[data_to_pick_up].values.tolist()
        token_id = 0
        tokens = []
        for datum in selected_data_to_choose_from:
            speaker, lang, surface = datum
            new_token = Token(surface=surface, lang=lang, speaker=speaker)
            tokens.append(new_token)
            token_id += 1

        list_of_collected_utterances.append(Utterance(tokens=tokens, speaker=speaker))

    return list_of_collected_utterances


def test_collect_all_utterances():
    file_number = 0
    root_dir = os.path.join(os.getcwd(), 'Data', 'bangor_raw')
    filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    for filename in filenames:
        if filename.endswith('.csv'):
            print("Current file = " + filename)
            print("File # = {}".format(file_number))
            list_of_utterances = collect_all_utterances(root_dir, filename)
            print("has {} utterances".format(len(list_of_utterances)))
            file_number += 1


def aggregate_utterances_to_turns(all_utterances_from_a_dialogue):
    list_of_turns = []
    current_speaker = None
    previous_speaker = None
    list_of_utterances_for_a_turn = []
    for utterance in all_utterances_from_a_dialogue:
        current_speaker = utterance.speaker

        if previous_speaker is None:  # open a new turn (first turn)
            list_of_utterances_for_a_turn.append(utterance)

        elif current_speaker == previous_speaker:  # Same speaker, Continuation of the turn
            list_of_utterances_for_a_turn.append(utterance)

        else:  # New Speaker
            list_of_turns.append(Turn(list_of_utterances_for_a_turn, speaker=previous_speaker))
            list_of_utterances_for_a_turn = [utterance]

        previous_speaker = current_speaker

    list_of_turns.append(Turn(list_of_utterances_for_a_turn, current_speaker))
    return list_of_turns


def test_aggregate_utterances_to_turns():
    root_dir = os.path.join(os.getcwd(), 'Data', 'bangor_raw')
    filename = 'herring1.csv'
    list_of_utterances = collect_all_utterances(root_dir, filename)
    list_of_turns = aggregate_utterances_to_turns(list_of_utterances)
    print(filename + ' has {} turns'.format(len(list_of_turns)))


def collect_list_of_speakers(turns):
    list_of_speakers = []
    for t in turns:
        if t.speaker not in list_of_speakers:
            list_of_speakers.append(t.speaker)
    return list_of_speakers


def test_collect_list_of_speakers():
    root_dir = os.path.join(os.getcwd(), 'Data', 'bangor_raw')
    filename = 'herring1.csv'
    list_of_utterances = collect_all_utterances(root_dir, filename)
    list_of_turns = aggregate_utterances_to_turns(list_of_utterances)
    list_of_speakers = collect_list_of_speakers(list_of_turns)
    print(filename + ' has {} turns'.format(len(list_of_turns)))
    print('list of speakers include:')
    print(list_of_speakers)


def collect_dialogue(root_dir, filename):
    utterances = collect_all_utterances(root_dir, filename)
    turns = aggregate_utterances_to_turns(utterances)
    list_of_speakers = collect_list_of_speakers(turns)
    return Dialogue(name=filename, turns=turns, utterances=utterances, list_of_speakers=list_of_speakers)


def test_collect_dialogue():
    root_dir = os.path.join(os.getcwd(), 'Data', 'bangor_raw')
    filename = 'herring1.csv'
    dialogue = collect_dialogue(root_dir, filename)
    print(dialogue)


def collect_corpus(corpus_name, root_dir):
    file_number = 0
    filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    dialogues = []
    for filename in filenames:
        if filename.endswith('.csv'):
            print("Current file = " + filename)
            print("File # = {}".format(file_number))
            new_dialogue = collect_dialogue(root_dir, filename)
            dialogues.append(new_dialogue)
            file_number += 1

    return Corpus(name=corpus_name, dialogues=dialogues)


def test_collect_corpus():
    corpus_name = 'Miami-Bangor Reduced'
    root_dir = os.path.join(os.getcwd(), 'Data', 'bangor_raw')
    c1 = collect_corpus(corpus_name, root_dir)
    print(c1)


def run_tests():
    # test_collect_all_utterances()
    # test_aggregate_utterances_to_turns()
    # test_collect_list_of_speakers()
    # test_collect_dialogue()
    test_collect_corpus()
    print("All tests finished successfully!")


if __name__ == '__main__':
    run_tests()
    print("success!")