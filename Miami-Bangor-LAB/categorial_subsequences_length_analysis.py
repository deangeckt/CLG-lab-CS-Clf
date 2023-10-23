import os.path
from collections import Counter
import random
import matplotlib.pyplot as plt
from scipy.special import zeta
import numpy as np
from utils import *

def subsequences_lengths_extractor(tags_sequence_extracted):
    tags_sequence_squoshed = []
    i = 0
    current_subsequence_length = 0
    prev_tag = None
    while i < len(tags_sequence_extracted):
        curr_tag = tags_sequence_extracted[i]
        if (prev_tag is None) or (curr_tag == prev_tag):
            current_subsequence_length += 1
        else:
            tags_sequence_squoshed.append((prev_tag, current_subsequence_length))
            current_subsequence_length = 1
        prev_tag = curr_tag
        i += 1

    if not(current_subsequence_length == 0):
        tags_sequence_squoshed.append((prev_tag, current_subsequence_length))
    return tags_sequence_squoshed


def test_subsequences_lengths_extractor():
    tags_sequence_extracted = ['eng', 'eng', 'spa',  'eng', 'heb', 'heb', 'heb', 'spa']
    print(tags_sequence_extracted)
    tags_sequence_squoshed = subsequences_lengths_extractor(tags_sequence_extracted)
    print(tags_sequence_squoshed)


def collect_tag_types(tags_sequence_squoshed):
    tag_types = []
    for (tag, subsequence_length) in tags_sequence_squoshed:
        if tag not in tag_types:
            tag_types.append(tag)
    return tag_types


def test_collect_tag_types():
    tags_sequence_extracted = ['eng', 'eng', 'spa', 'eng', 'heb', 'heb', 'heb', 'spa']
    print(tags_sequence_extracted)
    tags_sequence_squoshed = subsequences_lengths_extractor(tags_sequence_extracted)
    print(tags_sequence_squoshed)
    tag_types = collect_tag_types(tags_sequence_squoshed)
    print(tag_types)


def convert_to_frequency_vector(frequency_counter):
    max_key = max(frequency_counter.keys())
    frequency_vector = [0 for _ in range(max_key+1)]
    for i, frequency in frequency_counter.items():
        frequency_vector[i] = frequency
    return frequency_vector


def test_convert_to_frequency_vector():
    frequency_counter = Counter()
    frequency_counter.update([1, 2, 1, 2, 3])
    convert_to_frequency_vector(frequency_counter)


def collect_subsequence_frequencies(tags_sequence_extracted):
    tags_sequence_squoshed = subsequences_lengths_extractor(tags_sequence_extracted)
    tag_types = collect_tag_types(tags_sequence_squoshed)

    # Init:
    tags_subsequence_length_frequency_counters = {}
    for tag in tag_types:
        tags_subsequence_length_frequency_counters[tag] = Counter()

    # Collect Frequency Counters:
    for (tag, subsequence_length) in tags_sequence_squoshed:
        tags_subsequence_length_frequency_counters[tag].update([subsequence_length])

    # Convert to frequency vectors:
    tags_subsequence_length_frequency_vectors = {}
    for tag in tag_types:
        tags_subsequence_length_frequency_vectors[tag] = \
            convert_to_frequency_vector(tags_subsequence_length_frequency_counters[tag])

    return tags_subsequence_length_frequency_vectors


def test_collect_subsequence_frequencies():
    tags_sequence_extracted = ['eng', 'eng', 'spa', 'eng', 'spa', 'eng', 'spa', 'spa']
    tags_subsequence_length_frequency = collect_subsequence_frequencies(tags_sequence_extracted)
    print(tags_subsequence_length_frequency)


def pad_with_zeros(input_vector, required_length):
    padded_vector = input_vector.copy()
    if len(input_vector) < required_length:
        padded_vector = [0 for _ in range(required_length)]
        for i in range(len(input_vector)):
            padded_vector[i] = input_vector[i]
    return padded_vector


def test_pad_with_zeros():
    input_vector = [0, 3, 4, 5]
    required_length = 8
    padded_vector = pad_with_zeros(input_vector, required_length)
    print(padded_vector)

    input_vector = [0, 3, 4, 5, 0]
    required_length = 2
    padded_vector = pad_with_zeros(input_vector, required_length)
    print(padded_vector)


def unite_subsequence_frequencies(tags_subsequence_length_frequency1, tags_subsequence_length_frequency2):
    united_tags_subsequence_length_frequency = tags_subsequence_length_frequency1.copy()

    lang_tags1 = tags_subsequence_length_frequency1.keys()
    lang_tags2 = tags_subsequence_length_frequency2.keys()
    for lang_tag in lang_tags2:
        if lang_tag in lang_tags1:
            f1 = tags_subsequence_length_frequency1[lang_tag]
            f2 = tags_subsequence_length_frequency2[lang_tag]
            max_len = max(len(f1), len(f2))
            f1_padded = pad_with_zeros(f1, required_length=max_len)
            f2_padded = pad_with_zeros(f2, required_length=max_len)
            united_tags_subsequence_length_frequency[lang_tag] = pad_with_zeros([], required_length=max_len)
            for i in range(max_len):
                united_tags_subsequence_length_frequency[lang_tag][i] = f1_padded[i]+f2_padded[i]
        else:
            united_tags_subsequence_length_frequency[lang_tag] = tags_subsequence_length_frequency2[lang_tag]

    return united_tags_subsequence_length_frequency


def test_unite_subsequence_frequencies():
    tags_sequence_extracted1 = ['eng', 'eng', 'spa', 'eng', 'spa', 'eng', 'spa', 'spa']
    tags_subsequence_length_frequency1 = collect_subsequence_frequencies(tags_sequence_extracted1)
    tags_sequence_extracted2 = ['spa', 'spa', 'spa', 'eng', 'spa', 'eng', 'spa', 'spa']
    tags_subsequence_length_frequency2 = collect_subsequence_frequencies(tags_sequence_extracted2)
    united_tags_subsequence_length_frequency = \
        unite_subsequence_frequencies(tags_subsequence_length_frequency1, tags_subsequence_length_frequency2)
    print(united_tags_subsequence_length_frequency)


def calc_relative_frequency_of_tags(tags_subsequence_length_frequency):

    tags_subsequence_length_relative_frequency = {}

    tag_types = tags_subsequence_length_frequency.keys()

    for tag in tag_types:
        total_sum = sum(tags_subsequence_length_frequency[tag])

        # Filling In relative Frequency:
        if total_sum > 0:
            tags_subsequence_length_relative_frequency[tag] = [freq / total_sum for freq in
                                                               tags_subsequence_length_frequency[tag]]
        else:
            tags_subsequence_length_relative_frequency[tag] = [0]

    return tags_subsequence_length_relative_frequency


def test_calc_relative_frequency_of_tags():
    tags_sequence_extracted = ['eng', 'eng', 'spa', 'eng', 'spa']
    tags_sequence_extracted = random.choices(population=['eng', 'spa'], weights=[0.7, 0.3], k=5000)
    # print(tags_sequence_extracted)
    tags_subsequence_length_frequency = collect_subsequence_frequencies(tags_sequence_extracted)
    tags_subsequence_length_relative_frequency = calc_relative_frequency_of_tags(tags_subsequence_length_frequency)
    tag_types = tags_subsequence_length_frequency.keys()
    for tag in tag_types:
        print("TAG: " + tag)
        print(tags_subsequence_length_relative_frequency[tag])
        print(sum(tags_subsequence_length_relative_frequency[tag]))
        print('x'*3)
        plot_frequency(tags_subsequence_length_relative_frequency[tag])


def plot_frequency(histogram_values):
    x = [i for i in range(0, len(histogram_values))]
    plt.plot(x, histogram_values)
    plt.ylabel('some numbers')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title("Histogram")
    plt.show()
    input('waiting for any-key...')


def test_plot_frequency():
    histogram = [7, 3, 1, 0.5]
    plot_frequency(histogram)


def hazard_function_calculation(r):
    # Init:
    h = [0 for s in range(len(r))]

    # Assignment:
    for s in range(len(r)):
        den = sum([r[n] for n in range(len(r)) if n >= s])
        if den > 0:
            h[s] = r[s] / den
    return h


def test_hazard_function_calculation():
    r = [0.0, 0.8, 0.3, 0., 0.1, 0.05, 0., 0.]
    print(r)
    h = hazard_function_calculation(r)
    print(h)


def get_other_tag(tag_options, tag):
    if tag == tag_options[0]:
        the_other_tag = tag_options[1]
    elif tag == tag_options[1]:
        the_other_tag = tag_options[0]
    return the_other_tag


def test_get_other_tag():
    print('OTHER TAG:')
    tag_options = ['eng', 'spa']
    for tag in tag_options:
        other_tag = get_other_tag(tag_options, tag)
        print('tag: ' + tag)
        print('other tag: ' + other_tag)


def generate_random_sequence_from_hazard_functions(h1, h2, tag1, tag2, required_sequence_length=50):
    tag_options = [tag1, tag2]
    generated_sequence = [tag1]
    h = {}
    h[tag1] = h1
    h[tag2] = h2
    current_tag = tag1
    other_tag = tag2
    s = 1
    i = 0
    while i < required_sequence_length:
        """
        print('generated sequence:')
        print(generated_sequence)
        print("s = {}".format(s))
        print('h[current_tag]')
        print(h[current_tag])
        print(len(h[current_tag]))
        """
        p_change = h[current_tag][s]
        p_dont_change = 1-h[current_tag][s]
        next_tag = random.choices([current_tag, other_tag], weights=[p_dont_change, p_change], k=1)[0]
        generated_sequence.append(next_tag)

        if next_tag == current_tag:
            s += 1
        else:
            s = 1
        current_tag = next_tag
        other_tag = get_other_tag(tag_options=tag_options, tag=current_tag)
        i += 1

    return generated_sequence


def test_generate_random_sequence_from_hazard_function():
    tag1 = 'eng'
    tag2 = 'spa'

    r1 = [0.0, 0.8, 0.3, 0., 0.1, 0.05, 0., 0.]
    r2 = [0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]
    h1 = hazard_function_calculation(r1)

    h2 = hazard_function_calculation(r2)
    generated_sequence = generate_random_sequence_from_hazard_functions(h1, h2, tag1, tag2)
    print(generated_sequence)


def compare_histograms(h1, h2):
    x1 = [i for i in range(0, len(h1))]
    x2 = [i for i in range(0, len(h2))]

    plt.plot(x1, h1)
    plt.plot(x2, h2)

    plt.xlabel('sub-sequence length')
    plt.ylabel('Hazard function')
    plt.title("Hazard function Comparison")
    plt.show()
    input('waiting for any-key...')


def test_compare_histograms():
    r1 = [0.0, 0.8, 0.3, 0., 0.1, 0.05, 0., 0.]
    r2 = [0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]
    h1 = hazard_function_calculation(r1)
    h2 = hazard_function_calculation(r2)
    compare_histograms(h1, h2)


def compare_relative_frequencies(r1, r2):
    x1 = [i for i in range(0, len(r1))]
    x2 = [i for i in range(0, len(r2))]

    plt.plot(x1, r1)
    plt.plot(x2, r2)

    plt.xlabel('Sub-Sequence Length')
    plt.ylabel('Relative Frequency')
    plt.title("Relative-Frequency")
    plt.show()

    min_length = min(len(r1), len(r2))
    relative_frequency_difference = sum([abs(r1[i]-r2[i]) for i in range(min_length)])
    print("relative_frequency_difference = {}".format(relative_frequency_difference))
    input('waiting for any-key...')


def test_compare_relative_frequencies():
    r1 = [0.0, 0.8, 0.3, 0., 0.1, 0.05, 0., 0.]
    r2 = [0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]
    compare_relative_frequencies(r1, r2)


def test_histogram_reconstruction():
    generated_sequence_length = 1000000
    tag1 = 'eng'
    tag2 = 'spa'
    # r1 = [0.0, 0.8, 0.3, 0., 0.1, 0.05, 0., 0.]
    # r1 = [0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]
    # r2 = [0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]

    max_to_consider = 1000
    x1 = 1.4
    x2 = 2.1

    r1 = [1/(k**x1)/zeta(x1) for k in range(1, max_to_consider)]
    r2 = [1/(k ** x2)/zeta(x2) for k in range(1, max_to_consider)]

    r1.insert(0, 0)
    r2.insert(0, 0)

    h1 = hazard_function_calculation(r1)
    h2 = hazard_function_calculation(r2)

    generated_sequence = \
        generate_random_sequence_from_hazard_functions(h1, h2, tag1, tag2, required_sequence_length=generated_sequence_length)

    tags_subsequence_length_frequency = collect_subsequence_frequencies(generated_sequence)
    tag_types = tags_subsequence_length_frequency.keys()
    r1_reconstructed = tags_subsequence_length_frequency[tag1]
    h1_reconstructed = hazard_function_calculation(r1_reconstructed)
    compare_histograms(h1, h1_reconstructed)
    compare_histograms(r1, counter_to_relative_frequency(r1_reconstructed))

    r2_reconstructed = tags_subsequence_length_frequency[tag2]
    h2_reconstructed = hazard_function_calculation(r2_reconstructed)
    compare_histograms(h2, h2_reconstructed)
    compare_histograms(r2, counter_to_relative_frequency(r2_reconstructed))

    regenerated_sequence = \
        generate_random_sequence_from_hazard_functions(h1_reconstructed, h2_reconstructed, tag1, tag2, required_sequence_length=generated_sequence_length)

    tags_subsequence_length_frequency = collect_subsequence_frequencies(regenerated_sequence)
    tag_types = tags_subsequence_length_frequency.keys()
    r1_rereconstructed = tags_subsequence_length_frequency[tag1]
    h1_rereconstructed = hazard_function_calculation(r1_rereconstructed)
    # r11 = hazard_function_calculation(h1_rereconstructed)
    compare_histograms(h1_reconstructed, h1_rereconstructed)

    r2_rereconstructed = tags_subsequence_length_frequency[tag2]
    h2_rereconstructed = hazard_function_calculation(r2_rereconstructed)
    # r22 = hazard_function_calculation(h2_rereconstructed)
    compare_histograms(h2_reconstructed, h2_rereconstructed)

    # print(generated_sequence)
    # print(regenerated_sequence)
    changes_counter = 0
    for i in range(len(generated_sequence)):
        if not generated_sequence[i] == regenerated_sequence[i]:
            changes_counter += 1

    print("Total # of changes = {}".format(changes_counter))
    print("Relative % of changes = {}".format(100*changes_counter/len(generated_sequence)))


def hazard_to_relative_frequency(h):
    r = [0 for _ in range(len(h))]
    if len(h) > 0:
        r[0] = h[0]
        r_sum = r[0]
        for i in range(1, len(h)):
            r[i] = h[i] * (1-r_sum)
            r_sum += r[i]

    return r


def test_hazard_to_relative_frequency():
    r_original = [0.1, 0.35, 0.3, 0.15, 0.05, 0.05, 0., 0.]
    print(r_original)
    print(sum(r_original))
    h = hazard_function_calculation(r_original)
    print(h)
    print(sum(h))
    r_reconstrcuted = hazard_to_relative_frequency(h)
    print(r_reconstrcuted)
    print(sum(r_reconstrcuted))
    print([r_original[i]-r_reconstrcuted[i] for i in range(len(r_original))])


def counter_to_histogram(counter):
    max_key = max(counter.keys()) + 1
    h = [0 for _ in range(max_key)]
    for key, frequency in counter.items():
        h[key] = frequency
    return h


def test_counter_to_histogram():
    counter = Counter()
    tags_sequence_extracted = ['spa', 'eng', 'eng', 'spa', 'eng', 'spa', 'eng', 'spa', 'spa']
    tags_subsequence_length_frequency = collect_subsequence_frequencies(tags_sequence_extracted)
    print(tags_sequence_extracted)

    for tag in tags_subsequence_length_frequency.keys():
        print("Tag: " + tag)
        print(tags_subsequence_length_frequency[tag])
        h = counter_to_histogram(tags_subsequence_length_frequency[tag])

    print(h)


def counter_to_relative_frequency(counter):
    h = counter_to_histogram(counter)
    total_sum = sum(h)
    return [x/total_sum for x in h]


def test_counter_to_relative_frequency():
    counter = Counter()
    tags_sequence_extracted = ['spa', 'eng', 'eng', 'spa', 'eng', 'spa', 'eng', 'spa', 'spa']
    tags_subsequence_length_frequency = collect_subsequence_frequencies(tags_sequence_extracted)
    print(tags_sequence_extracted)
    for tag in tags_subsequence_length_frequency.keys():
        print("Tag: " + tag)
        print(tags_subsequence_length_frequency[tag])
        r = counter_to_relative_frequency(tags_subsequence_length_frequency[tag])
        print(r)
        print(sum(r))


def test_relative_frequency_reconstruction():
    generated_sequence_length = 1000000
    tag1 = 'eng'
    tag2 = 'spa'

    max_to_consider = 1000
    x1 = 1.4
    x2 = 2.1

    r1 = [1/(k**x1)/zeta(x1) for k in range(1, max_to_consider)]
    r2 = [1/(k ** x2)/zeta(x2) for k in range(1, max_to_consider)]

    r1.insert(0, 0)
    r2.insert(0, 0)

    h1 = hazard_function_calculation(r1)
    h2 = hazard_function_calculation(r2)

    generated_sequence = \
        generate_random_sequence_from_hazard_functions(h1, h2, tag1, tag2, required_sequence_length=generated_sequence_length)

    tags_subsequence_length_frequency = collect_subsequence_frequencies(generated_sequence)

    r1_reconstructed = tags_subsequence_length_frequency[tag1]
    h1_reconstructed = hazard_function_calculation(r1_reconstructed)
    compare_histograms(h1, h1_reconstructed)
    compare_relative_frequencies(r1, counter_to_relative_frequency(r1_reconstructed))

    r2_reconstructed = tags_subsequence_length_frequency[tag2]
    h2_reconstructed = hazard_function_calculation(r2_reconstructed)
    compare_histograms(h2, h2_reconstructed)
    compare_relative_frequencies(r2, counter_to_relative_frequency(r2_reconstructed))

    regenerated_sequence = \
        generate_random_sequence_from_hazard_functions(h1_reconstructed, h2_reconstructed, tag1, tag2, required_sequence_length=generated_sequence_length)

    tags_subsequence_length_frequency = collect_subsequence_frequencies(regenerated_sequence)
    r1_rereconstructed = tags_subsequence_length_frequency[tag1]
    h1_rereconstructed = hazard_function_calculation(r1_rereconstructed)
    compare_histograms(h1_reconstructed, h1_rereconstructed)

    compare_relative_frequencies(counter_to_relative_frequency(r1_reconstructed), counter_to_relative_frequency(r1_rereconstructed))

    r2_rereconstructed = tags_subsequence_length_frequency[tag2]
    h2_rereconstructed = hazard_function_calculation(r2_rereconstructed)

    compare_histograms(h2_reconstructed, h2_rereconstructed)
    compare_relative_frequencies(counter_to_relative_frequency(r2_reconstructed), counter_to_relative_frequency(r2_reconstructed))

    # print(generated_sequence)
    # print(regenerated_sequence)
    changes_counter = 0
    for i in range(len(generated_sequence)):
        if not generated_sequence[i] == regenerated_sequence[i]:
            changes_counter += 1

    print("Total # of changes = {}".format(changes_counter))
    print("Relative % of changes = {}".format(100*changes_counter/len(generated_sequence)))


def plot_relative_frequency_comparison(relative_frequencies_data, title='Relative Histogram Comparison', required_tags=['eng', 'spa']):
    data_vector_lengths = [len(relative_frequencies_data[tag]) for tag in required_tags]
    max_len = max(data_vector_lengths)
    indentation_index = -1
    indentation_value = 0
    for tag in required_tags:
        x0 = [i+indentation_index*indentation_value for i in range(1, max_len)]
        x1 = [np.log(x) for x in x0]
        y0 = pad_with_zeros(relative_frequencies_data[tag], max_len)[1:]
        y1 = [np.log(y) for y in y0]
        # plt.bar(x0, y0, 0.4, label=tag)
        plt.plot(x1, y1, '.', label=tag)
        indentation_index += 1

    plt.xlabel("log(s)")
    plt.ylabel("log(r)")
    plt.title(title)
    plt.legend()
    # plt.savefig(os.path.join(FOLDER_OF_FIGURES, title))
    plt.show()


def test_plot_relative_frequency_comparison():
    relative_frequencies_data = {}
    relative_frequencies_data['eng'] = [0, 0.4, 0.3, 0.2, 0.1]
    relative_frequencies_data['spa'] = [0, 0.6, 0.2, 0.1, 0.1]
    plot_relative_frequency_comparison(relative_frequencies_data, title='title', required_tags=['eng', 'spa'])


def run_tests():
    # test_convert_to_frequency_vector()
    # test_pad_with_zeros()
    # test_calc_relative_frequency_of_tags()
    # test_unite_subsequence_frequencies()
    # test_calc_relative_frequency_of_tags()
    # test_subsequences_lengths_extractor()
    # test_collect_tag_types()
    # test_collect_subsequence_frequencies()
    # test_calc_relative_frequency_of_tags()
    # test_plot_frequency()
    # test_hazard_function_calculation()
    # test_generate_random_sequence_from_hazard_function()
    # test_get_other_tag()
    # test_generate_random_sequence_from_hazard_function()
    # test_compare_histograms()
    # test_histogram_reconstruction()
    # test_hazard_to_relative_frequency()
    # test_counter_to_histogram()
    # test_counter_to_relative_frequency()
    # test_compare_relative_frequencies()
    # test_relative_frequency_reconstruction()
    test_plot_relative_frequency_comparison()

if __name__ == '__main__':
    run_tests()
    print("Success!")
