from fractions import Fraction
import pandas as pd
import os

# Reads the data from the same directory as this file of python code.
current_file_directory = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(current_file_directory, 'PlayTennis.csv'))

HYPHEN_LINE = '------------------------------------'

# function to calculate likelihood probabilities
def calc_likelihood_probs(data):
    likelihoods = {}
    # split data into yes and no groups based on play tennis column
    yes_group = data[data['Play Tennis'] == 'Yes']
    no_group = data[data['Play Tennis'] == 'No']
    classes = {'Yes': yes_group, 'No': no_group}

    # calculate likelihood probabilities for each class group
    for class_label, group in classes.items():
        feature_likelihoods = {}
        # calculate likelihood probabilities for each feature
        for feature in group.columns[:-1]:
            value_likelihoods = {}
            # calculate likelihood for each value of the feature
            for value in group[feature]:
                value_count = len(group[group[feature] == value])
                value_likelihoods[value] = Fraction(value_count, len(group)).limit_denominator() 
            feature_likelihoods[feature] = value_likelihoods
        likelihoods[class_label] = feature_likelihoods

    return likelihoods


# calculate likelihood probabilities
likelihood_probs = calc_likelihood_probs(data)

# display likelihood probabilities
print(HYPHEN_LINE)
for feature, values in likelihood_probs['Yes'].items():
    df = pd.DataFrame({'Yes': values, 'No': likelihood_probs['No'][feature]}, columns=likelihood_probs.keys(), index=values.keys())
    df.columns.name = feature
    df = df.fillna(0)
    df = df.sort_index(axis=0)
    print(f"{df}\n{HYPHEN_LINE}")

# function to calculate prior probabilities
def calc_prior_probs(data):
    total_count = len(data)
    class_labels = data['Play Tennis'].unique()
    prior_probs = {}
    # calculate prior probabilities for each class label
    for class_label in class_labels:
        class_group = data[data['Play Tennis'] == class_label]
        class_count = len(class_group)
        prior_probs[class_label] = class_count / total_count
    return prior_probs

# calculate prior probabilities
prior_probs = calc_prior_probs(data)

def predict(test_sample, likelihood_probs, prior_probs):
    # Initialize probabilities for both classes
    yes_prob = prior_probs['Yes']
    no_prob = prior_probs['No']
    
    # Multiply the probabilities of each feature value for both classes
    for feature, value in test_sample.items():
        if value in likelihood_probs['Yes'][feature]:
            yes_prob *= likelihood_probs['Yes'][feature][value]
        else:
            yes_prob *= 0
        if value in likelihood_probs['No'][feature]:
            no_prob *= likelihood_probs['No'][feature][value]
        else:
            no_prob *= 0
        print(f"{HYPHEN_LINE}\nP({feature}={value}|Play=Yes) = {likelihood_probs['Yes'][feature][value]}")
        print(f"P({feature}={value}|Play=No) = {likelihood_probs['No'][feature][value]}")
    print(f"{HYPHEN_LINE}\nP(Play=Yes) = {Fraction(prior_probs['Yes']).limit_denominator()} \nP(Play=No) = {Fraction(prior_probs['No']).limit_denominator()}")
    print(f"{HYPHEN_LINE}\nP(Yes|X`) = {round(yes_prob, 4)} \nP(No|X`) = {round(no_prob, 4)}\n{HYPHEN_LINE}")
    return 'Yes' if yes_prob > no_prob else 'No'

# Test the model with a sample test data
test_sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
print("\nX`=", test_sample, "\n")
predicted_class = predict(test_sample, likelihood_probs, prior_probs)
print("Predicted class of X`:", predicted_class)

