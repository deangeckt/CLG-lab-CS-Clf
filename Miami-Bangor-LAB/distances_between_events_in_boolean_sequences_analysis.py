import os.path
import random
import matplotlib.pyplot as plt
from utils import *


def extract_distances(boolean_series):
    indices_where_true = [i for i in range(len(boolean_series)) if boolean_series[i]]
    distances_between_true = [indices_where_true[j+1]-indices_where_true[j] for j in range(len(indices_where_true)-1)]
    return distances_between_true


def test_extract_distances():
    boolean_series = [True, False, True, True, False, True]
    distances_between_true = extract_distances(boolean_series)


def calc_frequency(distances):
    if len(distances) == 0:
        f = []
    else:
        max_distance = max(distances)
        f = [0 for _ in range(max_distance + 1)]

        for d in distances:
            f[d] += 1

    return f


def test_calc_frequency():
    d = [1, 2, 1, 3, 1]
    f = calc_frequency(d)
    print(f)


def calc_relative_frequency(frequency_counter):
    total_sum = sum(frequency_counter)
    r = [frequency_counter[d] / total_sum for d in range(len(frequency_counter))]
    return r


def test_calc_relative_frequency():
    print("test_calc_relative_frequency:")
    distances = [1, 1, 3]
    print(distances)
    f = calc_frequency(distances=distances)
    r = calc_relative_frequency(frequency_counter=f)
    print(r)


def calc_hazards(r):
    h = [0 for _ in range(len(r))]
    for n in range(len(h)):
        den = sum([r[k] for k in range(len(r)) if k >= n])
        num = r[n]
        h[n] = num/den
    return h


def test_calc_hazard():
    print("test_calc_hazard")
    r = [0.0, 0.6666666666666666, 0.0, 0.3333333333333333]
    print(r)
    h = calc_hazards(r)
    print(h)


def generate_series(h, n=10, d=1):
    s = []

    while len(s) <= n:
        if d >= len(h):
            print("Failure: Input distance should be less than maximal hazard")

        next_value = random.choices([True, False], [h[d], 1-h[d]])[0]
        s.append(next_value)

        if next_value:
            d = 1
        else:
            d += 1

    return s


def test_generate_series():
    print("test_generate_series")
    f = [0, 3, 2, 1, 0, 1]
    r = calc_relative_frequency(f)
    h = calc_hazards(r)
    s = generate_series(h, n=10, d=1)
    print(s)


def relative_frequency_comparison(r1, r2):
    x1 = [i for i in range(0, len(r1))]
    x2 = [i for i in range(0, len(r2))]

    plt.plot(x1, r1, '.')
    plt.plot(x2, r2, '.')

    plt.xlabel('Distances between True values')
    plt.ylabel('relative Frequency')
    plt.title("Relative Frequency Comparison")
    plt.show()

    diff = sum([abs(r1[i]-r2[i]) for i in range(min(len(r1), len(r2)))])
    print("The difference between the relative frequencies adds up to: {}".format(diff))
    input('waiting for any-key...')


def test_relative_frequency_comparison():
    r1 = [0, 4, 3, 2, 1, 0]
    r2 = [0, 5, 2, 3, 0, 1]
    relative_frequency_comparison(r1, r2)


def plot_relative_frequency(r, title="Relative Frequency Plot"):
    x = [i for i in range(1, len(r))]
    y = r[1:]  # without the 0 @ the 0 index

    # plt.plot(x, y, '.')
    plt.loglog(x, y, '.')

    plt.xlabel('Distances between True values')
    plt.ylabel('relative Frequency')
    plt.title(title)
    # plt.savefig(os.path.join(FOLDER_OF_FIGURES, title))
    plt.show()

    input('waiting for any-key...')


def test_plot_relative_frequency():
    f = [0, 4, 3, 2, 1, 0]
    r = calc_relative_frequency(f)
    plot_relative_frequency(r)


def run_tests():
    """
    test_extract_distances()
    test_calc_frequency()
    test_calc_relative_frequency()
    test_calc_hazard()
    test_generate_series()
    test_relative_frequency_comparison()
    """
    test_plot_relative_frequency()


def main():
    s0 = [True, False, False, True, True, False]
    d0 = extract_distances(s0)
    f0 = calc_frequency(d0)
    r0 = calc_relative_frequency(f0)
    h0 = calc_hazards(r0)
    s1 = generate_series(h0, n=100000, d=1)
    d1 = extract_distances(s1)
    f1 = calc_frequency(d1)
    r1 = calc_relative_frequency(f1)
    relative_frequency_comparison(r0, r1)


if __name__ == '__main__':
    run_tests()
    # main()