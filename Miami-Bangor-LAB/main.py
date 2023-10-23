from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from classes import *
from data_loaders import *
from products_generation import *
from data_analysis import *

MIAMI_BANGOR_CORPUS_NAME = "bm_tagged"
# MIAMI_BANGOR_CORPUS_ROOT_DIR = os.path.join(os.getcwd(), 'Data', 'bangor_raw_reduced')
MIAMI_BANGOR_CORPUS_ROOT_DIR = "bm_tagged" #os.path.join(os.getcwd(), 'Data', 'bangor_raw')
from google.cloud import translate_v2 as translate
import numpy as np
from sklearn.metrics import accuracy_score

def classify_lng_langid(corpus: Corpus):
    gt = []
    pred = []

    for dialogue in tqdm(corpus.dialogues):
        for utterance in dialogue.utterances:

            text = utterance.__str__()

            gt.append(utterance.lang)
            pred.append(langid_classify(text))
    print(accuracy_score(gt, pred))

    cm = confusion_matrix(gt, pred, labels=['eng', 'spa'], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['eng', 'spa'])
    disp.plot()
    plt.show()


def partition(lst, size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def classify_lng_google_detect(corpus: Corpus):
    gt = []
    pred = []
    # translate_client = translate.Client()
    all_texts = []
    for dialogue in tqdm(corpus.dialogues):
        for utterance in dialogue.utterances:
            text = utterance.__str__()
            all_texts.append(text)
            gt.append(utterance.lang)

    np.save('gt.npy', np.array(gt))

    pred = np.load('google_detect_pred.npy')


    # batchs = partition(all_texts, 100)
    # for b in batchs:
    #     detected = translate_client.detect_language(b)
    #     for d in detected:
    #         pred.append(d['language'])
    #
    # pred = np.array(pred)
    # np.save('google_detect_pred.npy', pred)

    pred = [p.replace('es', 'spa') for p in pred]
    pred = [p.replace('en', 'eng') for p in pred]

    print(accuracy_score(gt, pred))


    cm = confusion_matrix(gt, pred, labels=['eng', 'spa'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['eng', 'spa'])
    disp.plot()
    plt.show()


def main():
    miami_bangor_corpus = collect_corpus(corpus_name=MIAMI_BANGOR_CORPUS_NAME, root_dir=MIAMI_BANGOR_CORPUS_ROOT_DIR)
    # print(miami_bangor_corpus)
    classify_lng_langid(miami_bangor_corpus)
    classify_lng_google_detect(miami_bangor_corpus)
    # analyse_corpus(miami_bangor_corpus)
    # plot_cs(miami_bangor_corpus)



def pred_lang(text: str):
    confidence_values  = detector.compute_language_confidence_values(text)
    simple_list = [(v.value, v.language )for v in confidence_values]
    conf, lng = max(simple_list)
    print(conf, lng)

if __name__ == '__main__':
    # main()
    # print("Finished!")
    from lingua import Language, LanguageDetectorBuilder

    languages = [Language.ENGLISH, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    # print(detector.compute_language_confidence_values("Que"))
    # print(detector.detect_language_of("mas"))
    # print(detector.detect_language_of("At the second white dog. Que mas"))
    pred_lang('At the second white dog. Que mas')
