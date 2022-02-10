# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import random
from collections import defaultdict as defaultdict

# note to self: factoriser le code

def parse_sentence(lines):
    words = list()
    labels = list()
    # the expected word number corresponds
    # to the first number in word lines on the .conllu file
    word_number = 1
    for line in lines:
        columns = line.split("\t")
        if columns[0] == str(word_number):
            words.append(columns[1])
            labels.append(columns[3])
            word_number += 1
    return words, labels

'''  takes a filename as parameter and returns a list of examples. Each
example is made of pairs (observation, label) in which observation is a sentence (a
sequence of words) and label is a sequence of labels (there is one label for each word) '''
def corpus_reading(file_name):
    list_examples = list()
    file = open(file_name, "r")
    list_sentences = file.read().split("\n\n")
    for sentence in list_sentences:
        lines = sentence.split("\n")
        words, labels = parse_sentence(lines)
        if words:    # if sentence is not empty
            list_examples.append((words, labels))
    file.close()
    return list_examples

training_set = corpus_reading("/media/lina/RED/TP2ML/fr_gsd-ud-train.conllu")
#training_set = corpus_reading("D:\TP2ML\fr_gsd-ud-train.conllu")
print("the training set has " + str(len(training_set)) + " examples.")


label_dictionary = defaultdict(int)
for example in training_set:
    for label in example[1]:
        label_dictionary[label] += 1

'''
plt.title("Distribution of labels in the train set")

xpoints = list(label_dictionary.keys())
ypoints = list(label_dictionary.values())
plt.bar(xpoints,ypoints)
plt.show()
'''

test_set = corpus_reading("/media/lina/RED/TP2ML/fr_gsd-ud-test.conllu")
#test_set = corpus_reading("D:\TP2ML\fr_gsd-ud-test.conllu")
print("the test set has " + str(len(test_set)) + " examples.\n")


label_dictionary = defaultdict(int)
for example in training_set:
    for label in example[1]:
        label_dictionary[label] += 1

'''
plt.title("Distribution of labels in the test set")

xpoints = list(label_dictionary.keys())
ypoints = list(label_dictionary.values())
plt.bar(xpoints,ypoints)
plt.show()
'''

''' takes a corpus (list of sentences and their label) as input and return a list of pairs (feature vector, label) '''
def feature_extraction(list_observation):
    list_representation = list()
    for sentence, label in list_observation:
        for i in range(len(sentence)):
            feature_vector = list()
            feature_vector.append("curr_word_" + sentence[i])
            if i > 0:
                feature_vector.append("prev_word_" + sentence[i-1])
            if i > 1:
                feature_vector.append("prev_prev_word_" + sentence[i-2])
            if len(sentence) - i > 1:
                feature_vector.append("next_word_" + sentence[i+1])
            if len(sentence) - i > 2:
                feature_vector.append("next_next_word_" + sentence[i+2])
            feature_vector.append("biais")
            if sentence[i][0].isupper():
                feature_vector.append("starts_with_upper")
            if True in [char.isdigit() for char in sentence[i]]:
                feature_vector.append("contains_number")
            list_representation.append((feature_vector, label[i]))
    return list_representation

training_observations = feature_extraction(training_set)

''' auxiliary function used by the classifier
returns the sum of the weights in the parameter vector
of all features present in the observation vector '''
def dot_sparse(observation_vector, parameters_vector):
    return sum(parameters_vector.get(key, 0) for key in observation_vector)


class Perceptron:

    def __init__(self, labelset):
        ''' parametres est une liste contenant, pour chaque label,
        un tuple de la forme (label, parameter_vector_of_the_label) '''
        self.parameters = defaultdict(lambda: defaultdict(int))
        for label in labelset:
            self.parameters[label]

    ''' returns predicted label for observation '''
    def predict(self, observation):
        return max((dot_sparse(observation, self.parameters[label]), label) for label in self.parameters.keys())[1]

    def fit(self, training_observations, n_epochs, listener):
        for epoch in range(n_epochs):
            random.shuffle(training_observations)
            nb_mistakes = 0
            for observation, gold_label in training_observations:
                predicted_label = self.predict(observation)
                if predicted_label != gold_label:
                    nb_mistakes += 1
                    for feature in observation:
                        self.parameters[gold_label][feature] += 1
                        self.parameters[predicted_label][feature] -= 1
            listener(epoch, nb_mistakes)

    ''' takes a list of annotated observations of the form (features_vector, gold_label)
    and returns the proportion of correctly predicted labels (accuracy) '''
    def score(self, test_observations):
        return sum((self.predict(observation) == gold_label) for observation, gold_label in test_observations)*100/len(test_observations)


c = Perceptron(label_dictionary.keys())

def print_mistakes_when_training(epoch,nb_mistakes):
    print("epoch " + str(epoch + 1) + ": " + str(nb_mistakes) + " mistakes")

xpoints = list()
ypoints = list()
def plot_mistakes_when_training(epoch,nb_mistakes):
	xpoints.append(epoch)
	ypoints.append(nb_mistakes)

c.fit(training_observations, 10, plot_mistakes_when_training)

print("\naccuracy: " + str(c.score(feature_extraction(test_set))) + "%\n")

plt.title("Learning curve for the vanilla perceptron")
plt.xlabel("Epoch")
plt.ylabel("Number of mistakes made when classifying the training set")

plt.plot(xpoints, ypoints, marker = 'o')
plt.show()

'''
# testing on a small hand-made example
test_observations = feature_extraction([(["Le", "chat", "est", "triste", "."], ["DET", "NOUN", "VERB", "ADJ", "PUNCT"])])

for o,l in test_observations:
	print("predictated label: " + c.predict(o) + "\tgold label: " + l)
'''
