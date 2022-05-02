from conllu import parse, TokenList
from torch.utils.data import random_split
import torch
import csv
from flair.data import Corpus, Dictionary
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from sklearn.model_selection import train_test_split

def conllu_to_flair(path_in, lang, 
                    write_trainset: bool = False, 
                    write_raw: bool = False, 
                    write_testset: bool = False, 
                    write_dict: bool = False,
                    write_delex: bool = False):
    data = []
    raw_text = []
    dic = {}

    with open(path_in) as file:
        doc = file.read()
        doc = parse(doc)
        count: int = 0

        for line in doc:                           
            features = []
            raw = []
            count += 1

            for tok in line:
                if tok['upos'] != "_":
                    if write_delex == True:
                        tok['form'] = tok['upos']
                    else:
                        tok['form'] = tok['form'].replace("-", "")
                    combined = tok['form'] + " " + tok['upos'] + " " + str(tok['head']) + " " + tok['deprel']
                    features.append(combined)
                    raw.append(tok['form'])

            features = "\n".join(features)

            dic[count] = len(raw)
            raw = " ".join(raw)

            data.append(features)
            raw_text.append(raw)
    
    if write_dict == True:
        with open(f'data/{lang}/utterance_lengths.csv', 'w') as f:
            for key in dic.keys():
                f.write("%s, %s\n" % (key,dic[key]))      
        
    if write_trainset == True:
        train, validtext = train_test_split(data, random_state=42, test_size=.2)
        test, dev = train_test_split(validtext, random_state=42, test_size=0.5)

        all_in_one = "\n\n".join(data)
        dev = "\n\n".join(dev)
        test = "\n\n".join(test)
        train = "\n\n".join(train)
        
        data_folder = f'data/{lang}/features'
        
        with open(f'{data_folder}/dev.txt', 'w') as f:
            f.write(dev)
        with open(f'{data_folder}/test.txt', 'w') as f:
            f.write(test)
        with open(f'{data_folder}/train.txt', 'w') as f:
            f.write(train)
        with open(f'{data_folder}/all_in_one.txt', 'w') as f:
            f.write(all_in_one)
        
    if write_raw == True:
        raw_text = " ".join(raw_text)
        
        if write_delex == True:
            with open(f'data/{lang}/delex/corpus.txt', 'w') as f:
                f.write(raw_text)
        else:
            train, validtext = train_test_split(raw_text, random_state=42, test_size=.2)
            test, dev = train_test_split(validtext, random_state=42, test_size=0.5)

            data_emb = f'data/{lang}/embeddings'
            dev_raw = "\n\n".join(dev)
            test_raw = "\n\n".join(test)
            train_raw = "\n\n".join(train)

            with open(f'{data_emb}/char_lm/valid.txt', 'w') as f:
                f.write(dev_raw)
            with open(f'{data_emb}/char_lm/test.txt', 'w') as f:
                f.write(test_raw)
            with open(f'{data_emb}/char_lm/train/ud_train.txt', 'w') as f:
                f.write(train_raw)

            with open(f'{data_emb}/corpus.txt', 'w') as f:
                f.write(raw_text)

    if write_testset == True:
        lang_train, validtext = train_test_split(data, random_state=42, test_size=.4)
        lang_dev, lang_test = train_test_split(validtext, random_state=42, test_size=0.5)

        dev = "\n\n".join(lang_dev)
        test = "\n\n".join(lang_test)
        train = "\n\n".join(lang_train)
        
        data_folder = f'data/{lang}/features'

        with open(f'{data_folder}/dev_20.txt', 'w') as f:
            f.write(dev)
        with open(f'{data_folder}/test_20.txt', 'w') as f:
            f.write(test)
        with open(f'{data_folder}/train_60.txt', 'w') as f:
            f.write(train)
        
    print("Data successfully transformed. Have fun!") 

def make_trainset(language, reduced: bool = False):
    data_folder = f'data/{language}/features'
    columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}

    if reduced == True:
        corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file = 'train_60.txt',
                                test_file = 'test_20.txt',
                                dev_file = 'dev_20.txt')
    else:
        corpus: Corpus = ColumnCorpus(data_folder, columns,
                        train_file = 'train.txt',
                        test_file = 'test.txt',
                        dev_file = 'dev.txt')
    return corpus

def make_evalset(language):
    data_folder = f'data/{language}/features'
    columns = {0: 'text', 1: 'upos', 2:'head', 3: 'deprel'}

    corpus: Corpus = ColumnCorpus(data_folder, columns, test_file = 'all_in_one.txt')
    return corpus

def conllu_split(path_in, lang, write_trainset: bool = False, write_testset: bool = False, write_delex: bool = False):
    data = []
    raw_text = []
    count: int = 0

    with open(path_in) as file:
        doc = file.read()
        doc = parse(doc)

        for sentence in doc:
            utt = TokenList()

            for item in sentence:
                if item['upos'] != "_":
                    if write_delex == True:
                        item['form'] = item['upos']
                    else:
                        item['form'] = item['form'].replace("-", "")

                    if "nsubj" in item['deprel']:
                        item['deprel'] = "nsubj"
                    elif item['deprel'] == "aux:ev":
                        item['deprel'] = "aux:val"

                    utt.append(item)

            raw = utt.serialize()
            data.append(raw)

    if write_trainset == True:
        lang_train, validtext = train_test_split(data, random_state=42, test_size=.2)
        lang_val, lang_test = train_test_split(validtext, random_state=42, test_size=0.5)

        all_in_one = "".join(data)
        dev = "".join(lang_val)
        test = "".join(lang_test)
        train = "".join(lang_train)
        
        if write_delex == True:
            data_folder = f'data/{lang}/delex'
        else:
            data_folder = f'data/{lang}/conllu'

        with open(f'{data_folder}/dev.conllu', 'w') as f:
            f.write(dev)
        with open(f'{data_folder}/test.conllu', 'w') as f:
            f.write(test)
        with open(f'{data_folder}/train.conllu', 'w') as f:
            f.write(train)
        with open(f'{data_folder}/all_in_one.conllu', 'w') as f:
            f.write(all_in_one)

    if write_testset == True:
        lang_train, validtext = train_test_split(data, random_state=42, test_size=.4)
        lang_dev, lang_test = train_test_split(validtext, random_state=42, test_size=0.5)

        dev = "".join(lang_dev)
        test = "".join(lang_test)
        train = "".join(lang_train)

        if write_delex == True:
            data_folder = f'data/{lang}/delex'
        else:
            data_folder = f'data/{lang}/conllu'

        with open(f'{data_folder}/dev_20.conllu', 'w') as f:
            f.write(dev)
        with open(f'{data_folder}/test_20.conllu', 'w') as f:
            f.write(test)
        with open(f'{data_folder}/train_60.conllu', 'w') as f:
            f.write(train)

def concat_glove(lang):
    data_emb = f'data'

    lang_text = []
    with open(f'data/embeddings/train/monolingual.txt') as file:
        doc = file.read()
        text = doc.split("\n")
        for utt in text:
            lang_text.append(utt)

    with open(f'data/embeddings/train/parallel.txt') as file:
        doc = file.read()
        text = doc.split("\n")
        for utt in text:
            lang_text.append(utt)    
    
    with open(f'data/embeddings/train/monolingual.txt') as file:
        doc = file.read()
        text = doc.split("\n")
        for utt in text:
            lang_text.append(utt)

    with open(f'data/embeddings/valid.txt') as file:
        doc = file.read()
        text = doc.split("\n")
        for utt in text:
            lang_text.append(utt)

    with open(f'data/embeddings/test.txt') as file:
        doc = file.read()
        text = doc.split("\n")
        for utt in text:
            lang_text.append(utt)

    lang_text = " ".join(lang_text)

    with open(f'{data_emb}/corpus.txt', 'w') as f:
        f.write(lang_text)
