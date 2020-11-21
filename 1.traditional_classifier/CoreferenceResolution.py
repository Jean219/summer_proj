import sys
import numpy as np
import matplotlib.pyplot as plt

import FeatureExtractor
import CoreferenceClassifier

# The CoreferenceResolution class predicts the coreference probability of each document in GAP file, and outputs them into output_file.
class CoreferenceResolution:

    def __init__(self, feature_extractor, classifier, output_pred_filename) :

        processed_predicted_data = self._load_data(feature_extractor, classifier)
        self._output_file(processed_predicted_data, output_pred_filename)


    def _load_data(self, feature_extractor, classifier):
        predicted_data = []
        processed_predicted_data = []
        # document, mentions: [GAPSentence], [named_entity]
        for document, extracted_named_entities in feature_extractor.document_generator():
            mention_entity_pair = None
            mention_pair_pred = []

            for index, anaphor in enumerate(extracted_named_entities):
                for antecedent in reversed(extracted_named_entities[:index]):
                    if (anaphor[1] < antecedent[1]):
                        mention_entity_pair = (anaphor[1], antecedent[1])
                    else:
                        mention_entity_pair = (antecedent[1], anaphor[1])

                    # only when the mention_entity_pair contains pronoun_offset
                    if document.pronoun in mention_entity_pair:
                        mention_pair_feature_vector = feature_extractor.extract_feature_vector(antecedent, anaphor,
                                                                                               document)
                        pred = classifier.predict_mention_pair(mention_pair_feature_vector)

                        mention_pair_pred.append((mention_entity_pair,pred))

            # 以document为单位，批量process predicted_data至输出格式
            processed_data = self._process_predicted_data(document,mention_pair_pred)
            processed_predicted_data.append(processed_data)

        return processed_predicted_data


    def _process_predicted_data(self, document, mention_pair_pred):

        pa_prob = []
        pb_prob = []
        pother_prob = []
        for (mention_entity_pair,pred) in mention_pair_pred:
            if mention_entity_pair in [(document.pronoun, document.a),(document.a, document.pronoun)]:
                pa_prob.append(pred[0])
            elif mention_entity_pair in [(document.pronoun, document.b),(document.b, document.pronoun)]:
                pb_prob.append(pred[0])
            else:
                pother_prob.append(pred[0])

        # softmax the prob_a,prob_b,prob_other
        pa_prob = np.asarray(pa_prob).mean()
        pb_prob = np.asarray(pb_prob).mean()
        pother_prob = np.asarray(pother_prob).mean()

        abo_prob = np.asarray([pa_prob,pb_prob,pother_prob])
        # if abo_prob.sum() > 0:
        #     s_abo = softmax(abo_prob)
        # else:
        #     s_abo = np.asarray([1/3,1/3,1/3])
        s_abo = softmax(abo_prob)

        processed_predicted_data = (document.parag_id,document.pronoun,document.a,document.b,
                                    s_abo[0],s_abo[1],s_abo[2])
        return processed_predicted_data


    def _output_file(self, processed_predicted_data, output_pred_filename):

        pred_file = open(output_pred_filename, 'w')

        for row in processed_predicted_data:
            pred_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(row[0], # parag_id,
                                                                    row[1], # pronoun_offset
                                                                    row[2], # a_offset
                                                                    row[3], # b_offset
                                                                    row[4], # predicted_a_probability
                                                                    row[5],  # predicted_neither_probability
                                                                    row[6]))  # predicted_neither_probability

        pred_file.close()


def softmax(x):
    # Compute softmax values for each sets of scores in x
    e_x = np.exp(x)
    return e_x / e_x.sum()



if __name__ == '__main__' :
    # # ERROR: not right amount of arguments passed
    # if len(sys.argv) < 3 :
    #     print("Synopsis: FeatureExtractor.py INPUTFILE/FOLDER OUTPUTFILE (CLASSIFIER_BINARY)", file=sys.stderr)
    #     print("INPUTFILE is an OntoNote-File / FOLDER is a folder containing OntoNote-Files", file=sys.stderr)
    #     print("OUTPUTFILE is an OntoNote-File", file=sys.stderr)
    #     print("(CLASSIFIER_BINARY) is a binary containing an already trained model", file=sys.stderr)
    #     print("The dault classifier is SVM.", file=sys.stderr)
    #     sys.exit(1)
    #
    # else :
    #     path_to_test_instances = sys.argv[1]
    #     output_filename = sys.argv[2]
    #
    #     if len(sys.argv) == 4 :
    #         path_to_model = sys.argv[3]
    #     else :
    #         path_to_model = 'Trained_Models/SVM'
    #
    #     test_instances = FeatureExtractor.FeatureExtractor(path_to_test_instances)
    #     classifier = CoreferenceClassifier.CoreferenceClassifier.load_binary(path_to_model)
    #
    #     CoreferenceResolution(test_instances,classifier,output_filename)


    mode = 'DecisionTree'
    # mode = 'NaiveBayes'
    # mode = 'MaxEnt'
    # mode = 'RandomForest'

    # mode = 'SVM'  # 已删 AttributeError: probability estimates are not available for loss='hinge'
    # mode = 'Perceptron'  # 已删'Perceptron' object has no attribute 'predict_proba'
    # mode = 'LogisticRegression'  # 已删 从classifier就提示error


    # path_to_test_instances = '../data/gap-small.tsv'
    path_to_test_instances = '../data/gap-testSet.tsv'
    path_to_model = 'Trained_Models/' + mode
    output_filename = './performance_' + mode + '.data'

    feature_extractor = FeatureExtractor.FeatureExtractor(path_to_test_instances)

    classifier = CoreferenceClassifier.CoreferenceClassifier.load_binary(path_to_model)
    CoreferenceResolution(feature_extractor, classifier, output_filename)





