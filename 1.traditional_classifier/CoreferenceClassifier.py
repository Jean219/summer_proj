def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

import sys
import numpy
import pickle

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import FeatureExtractor

# The CoreferenceClassifier class takes mention-pairs with their features provided by FeatureExtractor
# and either learns from the mention-pairs a new model, or makes predictions about the coreference
# of this mention-pair.
class CoreferenceClassifier:

    def __init__(self, training_instances,classifier='DecisionTree') :

        if classifier not in {'NaiveBayes','Perceptron','MaxEnt','RandomForest','DecisionTree','LogisticRegression','__existing'} :
            print("ERROR: {} is not a valid classifier.".format(classifier),file=sys.stderr)
            print("Valid classifiers: ",file=sys.stderr)
            print("\t'NaiveBayes'\n\t'Perceptron'\n\t'MaxEnt'\n\t'RandomForest'\n\t'DecisionTree'\n\t'LogisticRegression'",file=sys.stderr)
            print("(Default = 'DecisionTree')",file=sys.stderr)
            sys.exit(1)

        # Classifier model
        self.classifier = classifier
        # Transformer that prepares data for training the model and making predictions
        self.column_transformer = None

        # leave constructor if loading an already trained model
        if classifier == '__existing' : #DecisionTree
            pass
        else :
            # Scaler and OneHotEncoder to adapt feature vectors to model
            self.column_transformer = ColumnTransformer([('NumericalData',StandardScaler(),[0]),
                                                         ('CategoricalData',OneHotEncoder(),slice(1,3))])

            self.column_transformer.fit([[0, '+', '+'],
                                         [0, '-', '-'],
                                         [0, 'unknown', 'unknown']])

            # incremental learning of model
            # if self.classifier == 'SVM' :
            #     self.classifier = SGDClassifier(loss='hinge')
            # if self.classifier == 'Perceptron' :
            #     self.classifier = Perceptron()
            if self.classifier == 'NaiveBayes' :
                self.classifier = BernoulliNB()
            elif self.classifier == 'MaxEnt' :
                self.classifier = SGDClassifier(loss='log')
            elif self.classifier == 'RandomForest' :
                self.classifier = RandomForestClassifier(warm_start=True)
            # elif self.classifier == 'LogisticRegression':
            #     self.classifier = LogisticRegression()
            elif self.classifier == 'DecisionTree': # this is just in case the default mode not work
                self.classifier = DecisionTreeClassifier(random_state=0)

            # over sampler to cope with uneven balanced class distributions
            # (there are a lot more non-coreferent mention-pairs than coreferent mention-pairs)
            over_sampler = RandomOverSampler()
            under_sampler = RandomUnderSampler()
            for parag in training_instances:
                feature_matrix = [x[2:5] for x in parag]
                labels = [x[5] for x in parag]
                if len(set(labels)) > 1 :
                    feature_matrix, labels = over_sampler.fit_resample(feature_matrix,labels)

                # update Scaler, using [x[0]:distance feature
                num_data = [[x[0]] for x in feature_matrix]
                self.column_transformer.named_transformers_['NumericalData'].partial_fit(num_data)
                del num_data

                # transform feature vectors
                feature_matrix = self.column_transformer.transform(feature_matrix)

                # update the model
                if classifier in ('RandomForest','DecisionTree','LogisticRegression') :
                    self.classifier.fit(feature_matrix,labels)
                else :
                    self.classifier.partial_fit(feature_matrix,labels,classes=['+','-','unknown'])

    # predict returns a vector containing the predicted classes for an input vector or matrix
    def predict(self,data) :
        # transform data so it fits the model
        try :
            data = self.column_transformer.transform(data)
        except ValueError :
            data = [data]
            data = self.column_transformer.transform(data)

        # make predictions
        # print("predict:",self.classifier.predict_proba(data)) ###################### add
        return self.classifier.predict_proba(data)

    def predict_mention_pair(self,feature_vector) :
        # transform feature vector so it fits the model
        feature_vector = self.column_transformer.transform([feature_vector])

        # make prediction
        pred = self.classifier.predict_proba(feature_vector)
        # print("predict_mention_pair:", pred[0])  ###################### add
        # return pred[0] # output the Label with the highest probability
        return pred # output all the probability of all the labels, here are 3 labels：+,-,unknown


    def transform(self,data) :
        return self.column_transformer.transform(data)

    # saves trained model in a binary file
    def save_binary(self,filename):
        binary = open(filename,'wb')
        pickle.dump((self.classifier,self.column_transformer),binary)
        binary.close()

    # reads a trained model from a binary file
    # Usage: classifier = CoreferenceClassifier.load_binary(filename)
    @classmethod
    def load_binary(cls, filename):
        classifier = CoreferenceClassifier([])
        binary = open(filename, 'rb')
        models = pickle.load(binary)
        binary.close()
        classifier.classifier = models[0]
        classifier.column_transformer = models[1]
        return classifier


if __name__ == '__main__':
    # # learning a new model and saving it as a binary for later use
    # # ERROR: not right amount of arguments passed
    # if len(sys.argv) < 3 :
    #     print("Synopsis: FeatureExtractor.py INPUTFILE/FOLDER OUTPUTFILE (CLASSIFIER)", file=sys.stderr)
    #     print("INPUTFILE/FOLDER is a OntoNote-File / FOLDER is a folder containing OntoNote-Files", file=sys.stderr)
    #     print("OUTPUTFILE is the model in binary format for later use", file=sys.stderr)
    #     print("CLASSIFIER is the algorithm that is used for learning (default=SVM)", file=sys.stderr)
    #     print("Possibilities:",file=sys.stderr)
    #     print("\tNaiveBayes")
    #     print("\tPerceptron")
    #     print("\tSVM")
    #     print("\tMaxEnt")
    #     print("\tRandomForest")
    #     sys.exit(1)
    #
    # else :
    #     path_to_training_instances = sys.argv[1]
    #     training_instances = FeatureExtractor.FeatureExtractor(path_to_training_instances).training_instances_iterator()
    #
    #     # train model
    #     # training algorithm is given
    #     if len(sys.argv) == 4 :
    #         mode = sys.argv[3]
    #         classifier = CoreferenceClassifier(training_instances, mode)
    #     # default training algorithm
    #     else :
    #         classifier = CoreferenceClassifier(training_instances)
    #
    #     # save model
    #     output = './Trained_Models/'+sys.argv[2]
    #     classifier.save_binary(output)

    # path_to_training_instances = '../data/gap-test.tsv'
    # path_to_training_instances = '../data/gap-validation.tsv'
    # path_to_training_instances = '../data/gap-development.tsv'
    # path_to_training_instances = '../data/gap-small.tsv'
    path_to_training_instances = '../gap-trainSet.tsv'

    training_instances = FeatureExtractor.FeatureExtractor(path_to_training_instances).training_instances_generator()

    # # train model, training algorithm is given
    mode = 'DecisionTree' # by default ok
    # mode = 'NaiveBayes'
    # mode = 'MaxEnt'
    # mode = 'RandomForest'
    # mode = 'SVM' #
    # mode = 'Perceptron'  #
    # mode = 'LogisticRegression'
    classifier = CoreferenceClassifier(training_instances, mode)

    # save model
    output = './Trained_Models/' + mode
    classifier.save_binary(output)

