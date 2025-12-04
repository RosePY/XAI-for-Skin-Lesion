from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

'''
CtL : Concept-to-Label

Class for the transparent model representing a function from concepts to task labels.
Represents the decision-making process of a given black-box model, in the concept representation
'''


class CtLModel:

    def __init__(self, c_data, y_data, **params):

        # Create copy of passed-in parameters
        self.params = params

        if 'method' in self.params:
            method = self.params["method"]
        else:
            method = 'DT'

        # Retrieve the classifier type
        self.clf_type = method

        # Retrieve total number of concepts
        self.n_concepts = c_data.shape[1]

        # Retrieve the concept names
        if "concept_names" in self.params:
            self.concept_names = self.params["concept_names"]
        else:
            self.concept_names = ["Concept " + str(i) for i in range(self.n_concepts)]

        # Retrieve the class names
        if "class_names" in self.params:
            self.class_names = self.params["class_names"]
        else:
            n_classes = np.max(y_data) + 1
            self.class_names = [str(i) for i in range(n_classes)]

        # Train classifier for predicting the output labels from concept data
        self.clf = self._train_label_classifier(c_data, y_data, self.clf_type)


    def _train_label_classifier(self, c_data, y_data, method='DT'):

        if method == 'DT':
            clf = DecisionTreeClassifier(random_state=0,class_weight='balanced', ccp_alpha = 0.0027)
        elif method == 'LR':
            clf = LogisticRegression(random_state=0,max_iter=200, class_weight='balanced', multi_class='auto', solver='lbfgs')
        elif method == 'LinearRegression':
            clf = LinearRegression()
        else:
            raise ValueError("Unrecognised model type...")

        clf.fit(c_data, y_data)

        return clf

    def get_sorted_coef(self):
        log_odds = self.clf.coef_[0]
        coef_df = pd.DataFrame(log_odds, 
             self.concept_names, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
        return coef_df

    def predict(self, c_data):
        return self.clf.predict(c_data)

