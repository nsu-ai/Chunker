import os
import pickle

from conllu import parse
from nltk.tokenize import sent_tokenize
from sklearn.externals import joblib
os.environ["KERAS_BACKEND"] = "tensorflow"
from deeppavlov import build_model, configs

linguistic_features = ['Abbr', 'AbsErgDatNumber', 'AbsErgDatPerson', 'AbsErgDatPolite', 'AdpType', 'AdvType', 'Animacy',
                       'Aspect', 'Case', 'Clusivity', 'ConjType', 'Definite', 'Degree', 'Echo', 'ErgDatGender',
                       'Evident', 'Foreign', 'Gender', 'Hyph', 'Mood', 'NameType', 'NounClass', 'NounType', 'NumForm',
                       'NumType', 'NumValue', 'Number', 'PartType', 'Person', 'Polarity', 'Polite', 'Poss', 'PossGender',
                       'PossNumber', 'PossPerson', 'PossedNumber', 'Prefix', 'PrepCase', 'PronType', 'PunctSide',
                       'PunctType', 'Reflex', 'Style', 'Subcat', 'Tense', 'Typo', 'VerbForm', 'VerbType', 'Voice', "Odd"]

class Chunker():
    def __init__(self, path_to_model, first_time=False):
        self.model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy, download=first_time)
        self._load_model(path_to_model)

    def _load_data(self, path_to_data):
        if os.path.isfile(path_to_data):
            file = open(path_to_data, encoding='utf-8').readlines()
            data = []
            for line in file:
                sentence = sent_tokenize(line)
                data += sentence
            data[0] = data[0].replace('\ufeff', '')
            return data
        else:
            raise IOError("The file {} does not exist!".format(path_to_data))

    def _get_morphotags(self, data):
        full_morphotags = []
        length = len(data)
        if length <= 100:
            morphotags = []
            for sent_parse in self.model(data):
                full_morphotags.append(sent_parse)
        else:
            times = length // 100
            rest = length % 100
            for n in range(times):
                from_ = n*10
                to_ = from_ + 100
                for sent_parse in self.model(data[from_:to_]):
                    full_morphotags.append(sent_parse)
            sent_rest = len(data)-rest
            for sent_parse in self.model(data[sent_rest:]):
                full_morphotags.append(sent_parse)
        list_of_sentences = []
        list_of_pos = []
        list_of_feat = []
        for word in full_morphotags:
            sent = parse(word)[0]
            new_list_words = []
            new_list_pos = []
            new_list_dict = []
            for word in sent:
                new_list_words.append(word['form'])
                new_list_pos.append(word['lemma'])
                features = word['upostag'].split('|')
                new_dict_feature = {}
                for feature in features:
                    feat = feature.split("=")
                    if len(feat) == 2:
                        new_dict_feature[feat[0]] = feat[1]
                    else:
                        new_dict_feature['Odd'] = '1'
                    for ling_feat in linguistic_features:
                        if ling_feat not in new_dict_feature:
                            new_dict_feature[ling_feat] = 0
                new_list_dict.append(new_dict_feature)

            assert len(new_list_pos) == len(new_list_words) == len(new_list_dict), "Error while pos-tagging!"
            list_of_sentences.append(new_list_words)
            list_of_pos.append(new_list_pos)
            list_of_feat.append(new_list_dict)
        return list_of_sentences, list_of_feat, list_of_pos

    def _create_X(self, list_of_sentences, list_of_feat, list_of_pos):
        X = []
        for num_sent, sentence in enumerate(list_of_sentences):
            if num_sent % 100 == 0:
                print("creating X: ", num_sent, '/', len(list_of_sentences))
            sentence_features = []
            for num_word, word in enumerate(sentence):
                tag = list_of_pos[num_sent][num_word]
                word_features = {
                    'token': word,
                    'tags': tag
                }
                word_features.update(list_of_feat[num_sent][num_word])
                sentence_features.append(word_features)
            X.append(sentence_features)
        X = [x for x in X if len(x) > 0]
        return X

    def _load_model(self, path_to_crf):
        if os.path.isfile(path_to_crf):
            self.crf = joblib.load(path_to_crf)
        else:
            raise IOError("The file {} does not exist!".format(path_to_crf))
        return self

    def _predict(self, X):
        return self.crf.predict(X)

    def predict_file(self, path_to_file):
        list_of_sentences, list_of_feat, list_of_pos = self._get_morphotags(self._load_data(path_to_file))
        X = self._create_X(list_of_sentences, list_of_feat, list_of_pos)
        predicted = self._predict(X)
        return predicted

    def predict_sentence(self, sentence):
        list_of_sentences, list_of_feat, list_of_pos = self._get_morphotags([sentence])
        X = self._create_X(list_of_sentences, list_of_feat, list_of_pos)
        predicted = self._predict(X)
        return predicted
