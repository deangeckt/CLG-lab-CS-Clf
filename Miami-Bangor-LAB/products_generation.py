from classes import *
from data_loaders import *
MAX_NUM_OF_UTTERANCES_TO_CONSIDER = 10


def plot_cs(corpus):
    print("corpus: " + corpus.name)
    printed_utterances_left = 0
    for dialogue in corpus.dialogues:

        for utterance in dialogue.utterances:
            if utterance.contains_intra_sentential_cs:
                printed_utterances_left = MAX_NUM_OF_UTTERANCES_TO_CONSIDER
                print("X" * 3 + '\n')
                print("dialogue: " + dialogue.name)
                print(utterance.speaker + ": (" + utterance.lang + ", CS) " + str(utterance))
            elif printed_utterances_left > 0:
                print(utterance.speaker + ": (" + utterance.lang + ") " + str(utterance))
                printed_utterances_left -= 1
        print("*"*10)


def test_plot_cs():
    MIAMI_BANGOR_CORPUS_NAME = "Miami-Bangor"
    MIAMI_BANGOR_CORPUS_ROOT_DIR = os.path.join(os.getcwd(), 'Data', 'bangor_raw')
    miami_bangor_corpus = collect_corpus(corpus_name=MIAMI_BANGOR_CORPUS_NAME,
                                             root_dir=MIAMI_BANGOR_CORPUS_ROOT_DIR)
    plot_cs(miami_bangor_corpus)


def run_tests():
    test_plot_cs()


if __name__ == '__main__':
    run_tests()
    print("Finished!")