# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import random
from collections import defaultdict as defaultdict

# try to make functions less than 10 lines long (factoriser le code) + remove cpmments before I send it + plotting the evolution of the number of mistakes for each epoch

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
print("the training set has " + str(len(training_set)) + " examples.")


label_dictionary = defaultdict(int)
for example in training_set:
    for label in example[1]:
        label_dictionary[label] += 1

'''
plt.title("Distribution of labels in the train set")

print(list(label_dictionary.keys()))
print(list(label_dictionary.values()))
xpoints = list(label_dictionary.keys())
ypoints = list(label_dictionary.values())
plt.bar(xpoints,ypoints)
plt.show()
'''

test_set = corpus_reading("/media/lina/RED/TP2ML/fr_gsd-ud-test.conllu")
print("the test set has " + str(len(test_set)) + " examples.\n")

'''
label_dictionary = defaultdict(int)
for example in training_set:
    for label in example[1]:
        label_dictionary[label] += 1

plt.title("Distribution of labels in the test set")

print(list(label_dictionary.keys()))
print(list(label_dictionary.values()))
xpoints = list(label_dictionary.keys())
ypoints = list(label_dictionary.values())
plt.bar(xpoints,ypoints)
plt.show()
'''

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


class AveragedPerceptron:

    def __init__(self, labelset):
        ''' parametres est une liste contenant, pour chaque label,
        un tuple de la forme (label, parameter_vector_of_the_label) '''
        self.w_parameters = defaultdict(lambda: defaultdict(int))
        self.a_parameters = defaultdict(lambda: defaultdict(int))
        for label in labelset:
            self.w_parameters[label]
            self.a_parameters[label]

    ''' prediction function used when training the perceptron with the working vector '''
    def w_predict(self, observation):
        return max((dot_sparse(observation, self.w_parameters[label]), label) for label in self.w_parameters.keys())[1]

    def fit(self, training_observations, n_epochs, listener):
        ''' date_last_update: dict(label, feature) -> number of examples we had seen the last time we modified this value
        this makes the code more efficient because we don't need to increment a value on the counter for each feature
        when the predicted label is correct, instead we only modify the counter dictionnary when we make a mistake '''
        date_last_update = defaultdict(int)
        nb_seen_examples = 0
        for epoch in range(n_epochs):
            random.shuffle(training_observations)
            nb_mistakes = 0
            for observation, gold_label in training_observations:
                predicted_label = self.w_predict(observation)
                if predicted_label != gold_label:
                    nb_mistakes += 1
                    for feature in observation:
                        self.a_parameters[gold_label][feature] += self.w_parameters[gold_label][feature] * (nb_seen_examples - date_last_update[gold_label, feature])
                        self.a_parameters[predicted_label][feature] += self.w_parameters[predicted_label][feature] * (nb_seen_examples - date_last_update[predicted_label, feature])

                        self.w_parameters[gold_label][feature] += 1
                        self.w_parameters[predicted_label][feature] -= 1

                        date_last_update[gold_label, feature] = nb_seen_examples
                        date_last_update[predicted_label, feature] = nb_seen_examples
                nb_seen_examples += 1
            listener(epoch, nb_mistakes)
        for label, feature in date_last_update:
            self.a_parameters[label][feature] += self.w_parameters[label][feature] * (nb_seen_examples - date_last_update[label, feature])

    ''' prediction function used after training that uses the averaged vector '''
    def a_predict(self, observation):
        return max((dot_sparse(observation, self.a_parameters[label]), label) for label in self.w_parameters.keys())[1]

    ''' takes a list of annotated observations of the form (features_vector, gold_label)
    and returns the proportion of correctly predicted labels (accuracy) '''
    def score(self, test_observations):
        return sum((self.a_predict(observation) == gold_label) for observation, gold_label in test_observations)*100/len(test_observations)

c = AveragedPerceptron(label_dictionary.keys())

def print_mistakes_when_training(epoch,nb_mistakes):
    print("epoch " + str(epoch + 1) + ": " + str(nb_mistakes) + " mistakes")

xpoints = list()
ypoints = list()
def plot_mistakes_when_training(epoch,nb_mistakes):
	xpoints.append(epoch)
	ypoints.append(nb_mistakes)

c.fit(training_observations, 10, plot_mistakes_when_training)


print("\naccuracy: " + str(c.score(feature_extraction(test_set))) + "%\n")

plt.title("Learning curve for the averaged perceptron")
plt.xlabel("Epoch")
plt.ylabel("Number of mistakes made when classifying the training set")

plt.plot(xpoints, ypoints, marker = 'o')
plt.show()

'''
# testing on a small hand-made example
test_observations = feature_extraction([(["Le", "chat", "est", "triste", "."], ["DET", "NOUN", "VERB", "ADJ", "PUNCT"])])

for o,l in test_observations:
	print("predicted label: " + c.a_predict(o) + "\tgold label: " + l)
'''
