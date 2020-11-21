import sys
import os
from nltk.corpus import wordnet
import csv
import numpy as np

import GAP
import MentionPair

# import en_core_web_lg
# nlp = en_core_web_lg.load()
import en_core_web_sm
nlp = en_core_web_sm.load()

# The FeatureExtractor class runs over GAP files provided and extracts 9 features out of mention-pairs.
class FeatureExtractor:
    def __init__(self, input_path):
        self.gender_frequency = {}
        # Gender by Name, DATASET BY DEREK HOWARD
        with open('resources/name_gender.csv', 'r', encoding='UTF-8') as gender_file:
            gender_data = csv.reader(gender_file)
            # gender_data = csv.DictReader(gender_file)
            for row in gender_data:
                self.gender_frequency[row[0].lower()] = row[1]
        gender_file.close()

        for male in ['he','him','his']:
            self.gender_frequency[male] = 'M'
        for female in ['she','her','hers']:
            self.gender_frequency[female] = 'F'

        if os.path.isfile(input_path):
            gap = GAP.GAP()
            self.data = gap.dataset_2_documents(input_path)
        else:
            print("ERROR: Cannot process input data", file=sys.stderr)
            print("'{}' does not lead to a file".format(input_path), file=sys.stderr)
            sys.exit(1)

    # genetate a list of MentionPair objects, using the preprocessed GAP data
    def training_instances_generator(self):
        training_instances = []

        for document in self.data:  # GAPDocument type
            extracted_named_entities = document.named_entities
            parag_mention_pairs = self._extract_mention_pairs_with_features(extracted_named_entities, document)
            if len(parag_mention_pairs) > 0:
                training_instances.append(parag_mention_pairs)

        return training_instances

    # genetate a list of (document, named_entities), using the preprocessed GAP data
    def document_generator(self):
        documents = []

        for document in self.data:  # GAPDocument type
            extracted_named_entities = document.named_entities
            documents.append((document, extracted_named_entities))

        return documents

    # antecedent/anaphora are both mentions，one NER，contains attributes(sentence_id, string_of_ne)
    # return mention_pair features (without mention_i/j string)
    def extract_feature_vector(self, antecedent, anaphora, document):
        # expand NER features, from 2 to 5
        antecedent = self._extract_independent_features(antecedent, document)
        anaphora = self._extract_independent_features(anaphora, document)

        # pair features, each has attrbites below:
        # self.mention_i: str = feature_tuple[0]
        # self.mention_j: str = feature_tuple[1]
        # self.distance: int = feature_tuple[2]
        # self.semantic_class_agreement = feature_tuple[3]
        # self.gender_agreement = feature_tuple[4]
        # self.coreferent = feature_tuple[5]
        mention_pair = MentionPair.MentionPair(self._extract_features(antecedent,anaphora,document))
        mention_pair = mention_pair.as_array()

        # return feature vector(3 features) without mention strings and without coreference label
        return mention_pair[2:5]

    def _extract_independent_features(self, ne, document):
        ne_sem_class = self._extract_sem_class(self._chose_synset(ne[2]))
        ne_gender = self._extract_gender(ne, document)
        features = (ne_sem_class,ne_gender)
        # RETURN: 5-tuple with the following content at index:
        #  0 - sentence ID
        #  1 - index of String
        #  2 - NP String
        #  3 - semantic class string (None if no class found)
        #  4 - gender string
        return (ne + features)

    # _extract_sem_class() looks if a noun (given in form of A WordNet Synset) contains to one of the
    # following classes and returns the class:
    # 'person': 'male', 'female'
    def _extract_sem_class(self, synset):
        # check if synset is found
        if synset is None:
            return None

        extract_hypernyms = lambda s: s.hypernyms()
        semantic_classes = {'person', 'male', 'female'}

        # extract all hypernyms
        all_hypernyms = [hyper.name() for hyper in list(synset.closure(extract_hypernyms))]

        # return first class found (= most specific class)
        for hypernym in all_hypernyms:
            if hypernym.split('.')[0] in semantic_classes:
                return hypernym.split('.')[0]

        # class not found
        return None

    # extracts the gender of a PERSON
    def _extract_gender(self, ne, document):
        # first we try to use some heuristics to extract gender feature
        fem_designators = {'she', 'mrs.', 'miss', 'ms.', 'madam', 'lady' ,'her'}
        masc_designators = {'he', 'mr.', 'sir','him','his'}

        frequency = None
        try:
            index = ne[1]
            text = document.text
            word_span = text[max(0, index-5):min(len(text), index + 10)].split(' ')
            for word in word_span:  # see if the ne context have indicative designators
                if word.lower() in fem_designators:
                    return 'feminine'
                if word.lower() in masc_designators:
                    return 'masculine'
        except ValueError:
            pass

        ne_list = ne[2].split(' ')
        for ne_string in ne_list:
            # look up with which gender the NER is most often associated
            try:
                gender = self.gender_frequency[ne_string.lower()]
            except KeyError:
                return 'unknown'

            # index 0 is masculine
            if gender == 'M':
                return 'masculine'
            # index 1 is feminine
            elif gender == 'F':
                return 'feminine'

        # index 2 is neuter
        return 'neuter'

    # extracts features of two NPs and returns a tuple containing these features
    def _extract_features(self, ne_i, ne_j, document):
        return (ne_i[2],  # String of ne_i
                ne_j[2],  # String of ne_j
                self._extract_distance(ne_i, ne_j),
                self._extract_if_sem_class_agreement(ne_i, ne_j),
                self._extract_if_gender_agreement(ne_i, ne_j),
                self._extract_if_coreferent(ne_i, ne_j, document))

    def _extract_features_for_prediction(self, antecedent, anaphora, document):
        return (antecedent[2],  # String of antecedent
                anaphora[2],  # String of anaphora
                self._extract_distance(antecedent, anaphora),
                self._extract_if_sem_class_agreement(antecedent, anaphora),
                self._extract_if_gender_agreement(antecedent, anaphora),
                None)

    def _extract_mention_pairs_with_features(self, named_entities_list, document):
        labeled_nes = []
        for ne in named_entities_list:
            # ne_sentence_id = ne[0]
            # ne_idx = ne[1]
            ne_string = ne[2]

            ne_sem_class = self._extract_sem_class(self._chose_synset(ne_string))
            ne_gender = self._extract_gender(ne, document)
            features = (ne_sem_class, ne_gender)
            # in labeled_nes are 5-tuples with the following content at index:
            #  0 - sentence ID
            #  1 - index of String
            #  2 - NER String
            #  3 - semantic class string (None if no class found)
            #  4 - gender string
            labeled_nes.append(ne + features)

        mention_pairs = []
        # extract mention-pair features
        for index, anaphora in enumerate(labeled_nes):
            # for antecedent in labeled_nes[len(labeled_nes):index:-1]:
            for antecedent in reversed(labeled_nes[:index]):
                if anaphora[1] != antecedent[1]:
                    # Mention-Pairs are extracted as explained in Soon et al.'s paper
                    mention_pair = MentionPair.MentionPair(self._extract_features(antecedent,anaphora,document))
                    if len(mention_pair) > 0:
                        # in mention_pairs, each element contains:
                        # self.mention_i: str = feature_tuple[0]
                        # self.mention_j: str = feature_tuple[1]
                        # self.distance: int = feature_tuple[2]
                        # self.semantic_class_agreement = feature_tuple[3]
                        # self.gender_agreement = feature_tuple[4]
                        # self.coreferent = feature_tuple[5]

                        # store MentionPair to training_instances, as list of array type
                        mention_pairs.append(mention_pair.as_array())
                    if mention_pair.is_coreferent():
                        break
        return mention_pairs

    # return the distance (in sentences) of two NER
    def _extract_distance(self, ne_i, ne_j):
        return int(ne_i[0] - ne_j[0])

    # returns True if two NERs share a semantic class or, if both classes are unknown,
    # returns False if the semantic classes differ, returns unknown if both classes are unknown
    def _extract_if_sem_class_agreement(self, ne_i, ne_j):
        ne_i_sem_class = ne_i[3]
        ne_j_sem_class = ne_j[3]

        # both classes are unknown
        if (ne_i_sem_class is None) and (ne_j_sem_class is None):
            return 'unknown'
        elif (ne_i_sem_class is None) or (ne_j_sem_class is None):
            return False

        # check for agreement
        person_hyponyms = {'person', 'male', 'female'}
        # classes agree
        if (ne_i_sem_class in person_hyponyms and ne_j_sem_class in person_hyponyms):
            return True
        # classes do not agree
        return False

    # returns True if both NPs agree in gender, else false
    # returns unknown if gender of both NPs is unknown
    def _extract_if_gender_agreement(self, ne_i, ne_j):
        gender_ne_i = ne_i[4]
        gender_ne_j = ne_j[4]

        # if at least one gender feature is unknown, gender agreement is unknown
        if (gender_ne_i == 'unknown') or (gender_ne_j == 'unknown'):
            return 'unknown'

        # test if gender features are equal
        return gender_ne_i == gender_ne_j

    # _chose_synset() extracts the first (= most frequent) nominal synset for an NP-head
    def _chose_synset(self, ne_string):
        ne_list = ne_string.split(" ")
        for ne in ne_list:
            # if the ne is a pronoun, replace it by a specific name
            if ne.lower in ['his', 'he', 'him']:
                ne = 'matthew'
            elif ne.lower() in ['she', 'her', 'hers']:
                ne = 'maria'
            else:
                ne = ne.lower()

            for synset in wordnet.synsets(ne):
                if synset.name().split('.')[1] == 'n':
                    return synset

        return None


    # returns True if two NERs are coreferent
    def _extract_if_coreferent(self, ne_i, ne_j, document):
        label_true=document.coref_spans
        label_false=document.coref_spans_false
        # print('label_true:',label_true)
        # print('label_false:',label_false)

        try:
            for x in [(ne_i[1],ne_j[1]),(ne_j[1],ne_i[1])]:
                if x in label_true:
                    # print('true')
                    return True
                elif x in label_false:
                    # print('false')
                    return False
            # print('unknown')
            return 'unknown'

        except ValueError:
            pass

if __name__ == '__main__':
    # # ERROR: not right amount of arguments passed
    # if len(sys.argv) < 3:
    #     print("Synopsis: FeatureExtractor.py INPUTFILE/FOLDER OUTPUTFILE", file=sys.stderr)
    #     print("INPUTFILE is a OntoNote-File / FOLDER is a folder containing OntoNote-Files", file=sys.stderr)
    #     print("OUTPUTFILE is the file, where the feature matrix should be saved", file=sys.stderr)
    #     sys.exit(1)
    # else:
    #     pathname = sys.argv[1]
    #     feature_extractor = FeatureExtractor(pathname)
    #
    #     training_instances = feature_extractor.training_instances_generator()
    #     print('type of training_instances:', type(training_instances))
    #     print('length of training_instances:',len(training_instances))
    #     for i in range(len(training_instances)):
    #         # print(training_instances[i].as_array())
    #         print(np.array(training_instances))



    pathname = '../data/gap-trainSet.tsv'
    # pathname = './data/gap-small.tsv'
    feature_extractor = FeatureExtractor(pathname)
    training_instances = feature_extractor.training_instances_generator()

    print('length of training_instances:', len(training_instances))
    # for i in range(len(training_instances)):
    #     print(training_instances[i])

    # # check the processed MentionPair data
    # pred_file = open('trainSet_MentionPair', 'w')
    # # pred_file = open('small_MentionPair', 'w')
    # for i in range(len(training_instances)):
    #     for j in range(len(training_instances[i])):
    #         pred_file.write("{}\t\n".format(training_instances[i][j]))
    # pred_file.close()


