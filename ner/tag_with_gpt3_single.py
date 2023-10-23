import pickle
import openai
from tqdm import tqdm
from read_corpus import *

openai.api_type = "azure"
openai.api_base = "https://aoai-east-us-d365sales-research.openai.azure.com/"
openai.api_version = "2022-06-01-preview"
openai.api_key = "34fe152b731749c597872a54bae5abe9"

pr_per_token = 0.0001200 # $ per token or 0.1200/1K tokens
total_tokens = 278257
total_price = 33.3 # $


def apply_model(prompt_content):
    response = openai.Completion.create(
        engine="d365-sales-davinci003",
        prompt=prompt_content,
        temperature=0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    return response

def create_prompt(text):
    prompt = 'Classify the following english-spanish sentence, token by token with 1 of the following categories: Person, Location, Organization, Time, Product, Number, Other.\n'
    prompt += f'"{text}".\n'
    prompt += 'Classify in the following format:\n'
    prompt += '<token number>:<word>:<category>\n'
    return prompt

def tag_all_utterances(utterances):
    docs = utterances

    for idx, uter in enumerate(tqdm(docs)):
        text = uter.__str__()
        prompt = create_prompt(text)
        try:
            res = apply_model(prompt)['choices'][0]['text'].strip().lower()
        except:
            print(idx)
            continue
        res = res.split('\n')

        if len(res) == len(uter.tokens):
            for ti, word in enumerate(res):
                entity = word.split(':')[2]
                uter.tokens[ti].ner = entity
        else:
            print(idx)

def tag():
    with open("../corpus/cs_uters.dat", "rb") as f:
        utterances = pickle.load(f)
    tag_all_utterances(utterances)

    with open("../corpus/cs_uters_with_ner.dat", "wb") as f:
        pickle.dump(utterances, f)


def fix_missing():
    with open ("../corpus/fix.txt", "rb") as f:
        lines = f.readlines()

    fixes = []
    for line in lines:
        line = line.decode("utf-8")
        if line == '':
            continue
        spt = line.strip().split(']')
        fail = int(spt[-1].strip())
        fixes.append(fail)

    del lines
    del spt
    del fail
    del line

    with open("../corpus/cs_uters_with_ner_fix.dat", "rb") as f:
        utterances : List[Utterance] = pickle.load(f)

    for fix in fixes:
        if fix < 409:
            continue
        uter = utterances[fix]
        for t in uter.tokens:
            t.ner = 'other'
        print(uter)

    with open("../corpus/cs_uters_with_ner_fix.dat", "wb") as f:
        pickle.dump(utterances, f)


def explore():
    with open("../corpus/cs_uters_with_ner_fix.dat", "rb") as f:
        utterances : List[Utterance] = pickle.load(f)

    ner_count = 0
    token_count = 0
    for uter in utterances:
        for token in uter.tokens:
            token_count += 1
            if token.ner != 'other':
                print(f'{token.surface} - {token.ner} -> {uter}')
                ner_count += 1

    print((ner_count/token_count)*100)
    print(ner_count, token_count)
    print(len(utterances))

# fix_missing()
explore()