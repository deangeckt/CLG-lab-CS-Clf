import langid
langid.set_languages(['en', 'es'])

from distances_between_events_in_boolean_sequences_analysis import *
from categorial_subsequences_length_analysis import *
from utils import *
from data_loaders import *


LANGID_CODES = {'en': 'eng', 'es': 'spa'}


def collect_languages(corpus):
    all_found_languages = []
    for dialogue in corpus.dialogues:
        for utterance in dialogue.utterances:
            for token in utterance.tokens:
                lang = token.lang
                if (lang not in all_found_languages) and (lang in WELL_DEFINED_LANGUAGE_OPTIONS):
                    all_found_languages.append(lang)

    return all_found_languages


def test_collect_languages():
    MIAMI_BANGOR_CORPUS_NAME = "Miami-Bangor"
    MIAMI_BANGOR_CORPUS_ROOT_DIR = os.path.join(os.getcwd(), 'Data', 'bangor_raw_reduced')
    miami_bangor_corpus = collect_corpus(corpus_name=MIAMI_BANGOR_CORPUS_NAME,
                                             root_dir=MIAMI_BANGOR_CORPUS_ROOT_DIR)
    all_found_languages = collect_languages(miami_bangor_corpus)
    print(all_found_languages)


def analyse_corpus_for_intra_sentential_cs(corpus):
    distances_between_intra_sentential_cs = []
    for dialogue in corpus.dialogues:
        does_utterance_contain_intra_sentential_cs = [u.contains_intra_sentential_cs for u in dialogue.utterances]
        distances_between_utterances_with_intra_sentential_cs_in_current_dialogue = \
            extract_distances(does_utterance_contain_intra_sentential_cs)
        distances_between_intra_sentential_cs.extend(
            distances_between_utterances_with_intra_sentential_cs_in_current_dialogue)

    frequency_of_intra_sentential_cs = calc_frequency(distances_between_intra_sentential_cs)
    relative_frequency_of_intra_sentential_cs = calc_relative_frequency(frequency_of_intra_sentential_cs)
    plot_title = "Relative Frequency for " + corpus.name + " IntRA-Sentential CS distances"
    plot_relative_frequency(relative_frequency_of_intra_sentential_cs, title=plot_title)


def analyse_corpus_for_inter_sentential_cs(corpus):
    language_tag_types_in_corpus = []
    language_tags_subsequence_length_frequency_in_corpus = {}
    for dialogue in corpus.dialogues:
        language_tags_of_utterances = [utterance.lang for utterance in dialogue.utterances if utterance.lang in WELL_DEFINED_LANGUAGE_OPTIONS]

        language_tags_subsequence_length_frequency_in_current_dialogue = \
            collect_subsequence_frequencies(language_tags_of_utterances)

        language_tags_subsequence_length_frequency_in_corpus = \
            unite_subsequence_frequencies(language_tags_subsequence_length_frequency_in_corpus,
                                          language_tags_subsequence_length_frequency_in_current_dialogue)

    language_tags_subsequence_length_relative_frequency_in_corpus = calc_relative_frequency_of_tags(
        language_tags_subsequence_length_frequency_in_corpus)
    """
    for tag in language_tags_subsequence_length_frequency_in_corpus.keys():
        plot_title = "Relative Frequency for " + tag + " for IntER-Sentential CS"
        plot_relative_frequency(language_tags_subsequence_length_relative_frequency_in_corpus[tag], title=plot_title)
    """
    plot_relative_frequency_comparison(language_tags_subsequence_length_frequency_in_corpus,
                                       title='Relative Comparison in Corpus',
                                       required_tags=['eng', 'spa'])



def analyse_langid_results(corpus):
    lang_analysis_results = {}
    for l1 in ['eng', 'spa']:
        for l2 in ['eng', 'spa']:
            lang_analysis_results[l1, l2] = 0

    for dialogue in corpus.dialogues:
        for utterance in dialogue.utterances:
            utterance_as_text = str(utterance)
            major_lang = utterance.lang
            langid_identified_lang = langid_classify(utterance_as_text)
            lang_analysis_results[major_lang, langid_identified_lang] += 1
            if not(major_lang == langid_identified_lang):
                print("major: " + major_lang)
                print("langid: " + langid_identified_lang)
                print(utterance)
                print("*"*5)

    print(lang_analysis_results)


def test_analyse_langid_results():
    MIAMI_BANGOR_CORPUS_NAME = "Miami-Bangor-Reduced"
    MIAMI_BANGOR_CORPUS_ROOT_DIR = os.path.join(os.getcwd(), 'Data', 'bangor_raw')
    miami_bangor_corpus = collect_corpus(corpus_name=MIAMI_BANGOR_CORPUS_NAME,
                                         root_dir=MIAMI_BANGOR_CORPUS_ROOT_DIR)
    analyse_langid_results(miami_bangor_corpus)


def analyse_corpus(corpus):
    corpus.tag_all_cs()

    # distances between IntRA-Sentential CS
    analyse_corpus_for_intra_sentential_cs(corpus)

    # distances between IntER-Sentential CS
    analyse_corpus_for_inter_sentential_cs(corpus)

    # langid analysis:
    analyse_langid_results(corpus)


def test_analyse_corpus():
    MIAMI_BANGOR_CORPUS_NAME = "Miami-Bangor-Reduced"
    MIAMI_BANGOR_CORPUS_ROOT_DIR = os.path.join(os.getcwd(), 'Data', 'bangor_raw_reduced')
    miami_bangor_corpus = collect_corpus(corpus_name=MIAMI_BANGOR_CORPUS_NAME,
                                         root_dir=MIAMI_BANGOR_CORPUS_ROOT_DIR)
    analyse_corpus(miami_bangor_corpus)


def langid_classify(text):
    classifier_result = langid.classify(text)
    return LANGID_CODES[classifier_result[0]]



def test_langid_classify():
    # t1 = 'Hello to you all...'
    # r1 = langid_classify(t1)
    # print(r1)
    # t2 = 'Quién es el tonto que escribió todas estas tonterías.'
    # r2 = langid_classify(t2)
    # print(r2)


    t = 've al loro and then switch'
    r = langid_classify(t)
    print(3)



def run_tests():
    # test_collect_languages()
    test_langid_classify()
    test_analyse_langid_results()
    # test_analyse_corpus()


if __name__ == '__main__':
    run_tests()
    print("Finished!")