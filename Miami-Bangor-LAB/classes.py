#  version: 1.1
#  Last Updated: 9.11.2022
from language_analysis import *


class Token(object):
    def __init__(self, surface, lang, speaker='OSE'):
        self.surface = surface
        self.lang = lang
        self.speaker = speaker

    def __str__(self):
        s = "The token is: " + self.surface
        s += '\n'
        return s


def generate_token():
    return Token(surface='hi', lang='eng', speaker='NET')


def test_token():
    t1 = generate_token()
    print(t1)


class Utterance(object):
    def __init__(self, tokens, speaker):
        self.tokens = tokens
        self.speaker = speaker

        self.lang = None
        self.contains_intra_sentential_cs = None

    def __str__(self):
        s = ''
        for token in self.tokens:
            s += ' ' + str(token.surface)
        return s.strip()

    def set_lang(self, lang):
        self.lang = lang

    def set_contains_intra_sentential_cs(self, contains_intra_sentential_cs):
        self.contains_intra_sentential_cs = contains_intra_sentential_cs


def generate_utterance():
    """
    speaker = 'NET'
    t1 = Token(surface='hi', lang='eng', speaker=speaker)
    t2 = Token(surface='amigo', lang='spa', speaker=speaker)
    t3 = Token(surface='adios', lang='spa', speaker=speaker)
    tokens = [t1, t2, t3]
    """
    speaker = 'MOT'
    t1 = Token(surface='we', lang='eng', speaker=speaker)
    t2 = Token(surface='could', lang='eng', speaker=speaker)
    t3 = Token(surface='shabat', lang='heb', speaker=speaker)
    t4 = Token(surface='?', lang='999', speaker=speaker)
    tokens = [t1, t2, t3, t4]
    return Utterance(tokens=tokens, speaker=speaker)


def test_utterance():
    utterance = generate_utterance()
    print(utterance)


class Turn(object):
    def __init__(self, utterances, speaker):
        self.utterances = utterances
        self.speaker = speaker
        self.tokens = [token for utterance in self.utterances for token in utterance.tokens]

        self.lang = None

    def __str__(self):
        s = ''
        for utterance in self.utterances:
            s += self.speaker + ': ' + str(utterance) + '\n'
        return s

    def set_lang(self, lang):
        self.lang = lang


def generate_turn():
    speaker = 'NET'
    utterances = []
    t1 = Token(surface='hi', lang='eng', speaker=speaker)
    t2 = Token(surface='adios', lang='spa', speaker=speaker)
    t3 = Token(surface='.', lang='999', speaker=speaker)
    tokens = [t1, t2, t3]
    utterances.append(Utterance(tokens=tokens, speaker=speaker))
    t1 = Token(surface='How', lang='eng', speaker=speaker)
    t2 = Token(surface='are', lang='spa', speaker=speaker)
    t3 = Token(surface='you', lang='eng', speaker=speaker)
    t4 = Token(surface='?', lang='999', speaker=speaker)
    tokens = [t1, t2, t3, t4]
    utterances.append(Utterance(tokens=tokens, speaker=speaker))

    return Turn(utterances=utterances, speaker=speaker)


def test_turn():
    turn = generate_turn()
    print(turn)


class Dialogue(object):
    def __init__(self, name, turns, utterances, list_of_speakers):
        self.name = name
        self.turns = turns
        self.utterances = utterances
        self.list_of_speakers = list_of_speakers

    def __str__(self):
        s = ''
        for turn in self.turns:
            s += str(turn)
        return s


def generate_dialogue():
    all_utterances = []
    turns = []

    speaker1 = 'NET'
    utterances = []
    t1 = Token(surface='hi', lang='eng', speaker=speaker1)
    t2 = Token(surface='adios', lang='spa', speaker=speaker1)
    t3 = Token(surface='.', lang='999', speaker=speaker1)
    tokens = [t1, t2, t3]
    utterance1 = Utterance(tokens=tokens, speaker=speaker1)
    utterances.append(utterance1)
    all_utterances.append(utterance1)

    t1 = Token(surface='How', lang='eng', speaker=speaker1)
    t2 = Token(surface='are', lang='spa', speaker=speaker1)
    t3 = Token(surface='you', lang='eng', speaker=speaker1)
    t4 = Token(surface='?', lang='999', speaker=speaker1)
    tokens = [t1, t2, t3, t4]
    utterance2 = Utterance(tokens=tokens, speaker=speaker1)
    utterances.append(utterance2)
    all_utterances.append(utterance2)
    turn1 = Turn(utterances=utterances, speaker=speaker1)
    turns.append(turn1)

    speaker2 = 'ADV'
    utterances = []
    t1 = Token(surface='Hi', lang='eng', speaker=speaker2)
    t2 = Token(surface='there', lang='spa', speaker=speaker2)
    t3 = Token(surface='.', lang='999', speaker=speaker2)
    tokens = [t1, t2, t3]
    utterance3 = Utterance(tokens=tokens, speaker=speaker2)
    utterances.append(utterance3)
    all_utterances.append(utterance3)

    t1 = Token(surface='How', lang='eng', speaker=speaker2)
    t2 = Token(surface='do', lang='spa', speaker=speaker2)
    t3 = Token(surface='you', lang='spa&eng', speaker=speaker2)
    t4 = Token(surface='do', lang='eng', speaker=speaker2)
    t5 = Token(surface='?', lang='999', speaker=speaker2)
    tokens = [t1, t2, t3, t4, t5]
    utterance4 = Utterance(tokens=tokens, speaker=speaker1)
    utterances.append(utterance4)
    all_utterances.append(utterance4)
    turn2 = Turn(utterances=utterances, speaker=speaker2)
    turns.append(turn2)

    list_of_all_speakers = [speaker1, speaker2]
    return Dialogue(turns=turns, utterances=all_utterances, list_of_speakers=list_of_all_speakers)


def test_dialogue():
    dialogue = generate_dialogue()
    print(dialogue)


class Corpus(object):
    def __init__(self, name, dialogues):
        self.name = name
        self.dialogues = dialogues
        self.tag_all_cs()

    def __str__(self):
        s = "The corpus " + self.name + " has {} dialogues".format(len(self.dialogues))
        return s

    def tag_all_cs(self):
        for dialogue in self.dialogues:
            for utterance in dialogue.utterances:
                utterance_lang = find_major_language(utterance)
                utterance.set_lang(utterance_lang)

                contain_intra_sentential_cs = does_it_contain_intra_sentential_cs(utterance)
                utterance.set_contains_intra_sentential_cs(contain_intra_sentential_cs)

            for turn in dialogue.turns:
                turn_lang = find_major_language(turn)
                turn.set_lang(turn_lang)


def test_corpus():
    dialogue_name = 'artificial_dialogue'
    dialogues = []
    num_of_dialogues = 3
    for _ in range(num_of_dialogues):
        dialogues.append(generate_dialogue())

    corpus = Corpus(name=dialogue_name, dialogues=dialogues)
    print(corpus)


def run_tests():
    test_token()
    test_utterance()
    test_turn()
    test_dialogue()
    test_corpus()


if __name__ == '__main__':
    run_tests()
    print("Success!")