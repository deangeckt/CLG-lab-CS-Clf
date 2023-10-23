from classes import *
from utils import *
import langid


def init_language_labels_frequency():
    frequency_of_language_labels = {}
    for lang in WELL_DEFINED_LANGUAGE_OPTIONS:
        frequency_of_language_labels[lang] = 0
    return frequency_of_language_labels


def collect_language_labels_frequency(clean_utterance_language_labels):
    frequency_of_language_labels = init_language_labels_frequency()
    for li in clean_utterance_language_labels:
        if li in frequency_of_language_labels:
            frequency_of_language_labels[li] += 1
        else:  # li = Both/Undefined etc.
            for lang in frequency_of_language_labels:
                frequency_of_language_labels[lang] += 0.5

    return frequency_of_language_labels


def remove_lang_label_ambiguity(utterance_language_labels, verbose=False):
    # we assume that the the utterance_language_labels do not contain any punctuation marks!

    clean_utterance_language_labels = []  # init return value

    def find_first_well_defined_label_in(label_seq):
        for label in label_seq:
            if label in WELL_DEFINED_LANGUAGE_OPTIONS:
                return label
        return None

    def test_find_first_well_defined_label_in():
        label_seq = ['eng&spa', 'eng', 'spa']
        res = find_first_well_defined_label_in(label_seq)
        print(res)

    # 0) Are all labels ambiguous?
    # print([label not in WELL_DEFINED_LANGUAGE_OPTIONS for label in utterance_language_labels])
    if len([label not in WELL_DEFINED_LANGUAGE_OPTIONS for label in utterance_language_labels]) == 0:
        if verbose:
            print("All labels are ill-defined")

    elif min([label not in WELL_DEFINED_LANGUAGE_OPTIONS for label in utterance_language_labels]):
        if verbose:
            print("All labels are ambiguous")

    else:
        for ind in range(len(utterance_language_labels)):
            current_label = utterance_language_labels[ind]
            if current_label in WELL_DEFINED_LANGUAGE_OPTIONS:
                clean_utterance_language_labels.append(current_label)

            else:  # 1) Look back:
                sequence_before_reversed = utterance_language_labels[0:ind].copy()
                sequence_before_reversed.reverse()
                best_label_before = find_first_well_defined_label_in(sequence_before_reversed)
                if best_label_before is not None:  # Well defined language label was found before
                    clean_utterance_language_labels.append(best_label_before)

                else:  # 2) Look forward:
                    best_label_after = \
                        find_first_well_defined_label_in(utterance_language_labels[ind:])
                    clean_utterance_language_labels.append(best_label_after)

    return clean_utterance_language_labels


def test_remove_lang_label_ambiguity():
    #utterance_language_labels = ['eng', 'eng', 'spa']
    utterance_language_labels = ['eng', 'eng', 'eng&spa']
    # utterance_language_labels = ['eng&spa', 'Undefined', 'eng&spa']
    print(utterance_language_labels)
    clean_utterance_language_labels = remove_lang_label_ambiguity(utterance_language_labels)
    print(clean_utterance_language_labels)


def test_remove_lang_label_ambiguity():
    #utterance_language_labels = ['eng', 'eng', 'spa']
    utterance_language_labels = ['eng', 'eng', 'eng&spa']
    utterance_surface_tokens = ['I', 'love', 'toasts']
    # utterance_language_labels = ['eng&spa', 'Undefined', 'eng&spa']
    print(utterance_language_labels)
    clean_utterance_language_labels = remove_lang_label_ambiguity(utterance_language_labels)


def extract_language_labels(utterance_or_turn):
    language_labels = []  # default
    if type(utterance_or_turn).__name__ == 'Utterance':
        language_labels = [token.lang for token in utterance_or_turn.tokens]
    elif type(utterance_or_turn).__name__ == 'Turn':
        language_labels = [token.lang for utterance in utterance_or_turn.utterances for token in utterance.tokens]
    return language_labels


def test_extract_language_labels():
    utterance = generate_utterance()
    print(utterance)
    print(utterance)
    print(extract_language_labels(utterance))
    turn = generate_turn()
    print(turn)
    print(extract_language_labels(turn))


def find_major_language(utterance_or_turn, default_language_code='eng')\
        -> 'eng, spa, None':

    major_language = default_language_code  # default value to return

    language_labels = extract_language_labels(utterance_or_turn)

    clean_language_labels = remove_lang_label_ambiguity(language_labels)
    num_of_tokens = len(clean_language_labels)

    if num_of_tokens > 0:
        frequency_of_language_labels = collect_language_labels_frequency(clean_language_labels)
        major_language_sorted = sorted(frequency_of_language_labels, key=frequency_of_language_labels.get, reverse=True)
        max_used_language = major_language_sorted[0]
        second_used_language = major_language_sorted[1]

        # Note: Solves the case of equality (i.e. if # of tokens in both languages is equal) with random
        if frequency_of_language_labels[max_used_language] > frequency_of_language_labels[second_used_language]:
            major_language = max_used_language

    return major_language


def test_find_major_language():
    print("test_find_major_language started...")
    utterance = generate_utterance()
    print(utterance)
    major_lang = find_major_language(utterance)
    print("Major Language = " + major_lang)

    turn = generate_turn()
    print(turn)
    major_lang = find_major_language(turn)
    print("Major Language = " + major_lang)

    print("test_find_major_language ended!!")


def does_it_contain_intra_sentential_cs(utterance_or_turn) -> 'Boolean':
    contain_intra_sentential_cs = False

    language_labels = extract_language_labels(utterance_or_turn)
    unique_language_labels = [label for label in language_labels if label in WELL_DEFINED_LANGUAGE_OPTIONS]
    if len(set(unique_language_labels)) >= 2:
        contain_intra_sentential_cs = True

    return contain_intra_sentential_cs


def test_does_it_contain_intra_sentential_cs():
    print("test_does_it_contain_intra_sentential_cs...")
    utterance = generate_utterance()
    print(utterance)
    contain_intra_sentential_cs = does_it_contain_intra_sentential_cs(utterance)
    print("contain_intra_sentential_cs = " + str(contain_intra_sentential_cs))

    turn = generate_turn()
    print(turn)
    contain_intra_sentential_cs = does_it_contain_intra_sentential_cs(turn)
    print("contain_intra_sentential_cs = " + str(contain_intra_sentential_cs))

    print("test_does_it_contain_intra_sentential_cs!!")


def run_tests():
    test_remove_lang_label_ambiguity()
    test_extract_language_labels()
    test_find_major_language()
    test_does_it_contain_intra_sentential_cs()


if __name__ == '__main__':
    run_tests()
    print("Finished!")