from sklearn import svm
import taxonomie

class PredictionVectorzier():

    def __init__(self):
        self.clf = svm.SVC(kernel='linear', probability=True, C=1)

    def fit(self, X, y):

        self.clf.fit(X, y)

    def predict(self, X):

        pred_probs = self.clf.predict_proba(X)

        #predictions = self.clf.predict(X)

        #print(pred_probs)

        return pred_probs

    def label_transformer(self, mapping, y):
        '''
        transforms labels of level i into labels of level i+1 in the taxonomy
        :param mapping: list of labels [label_1, label_2]
        :param y: list of numerical labels  [0,1]
        :return: y transformed to top level labels
        '''

        tax = taxonomie.Taxonomie().tax
        y_new = []

        for label in y:
            label = mapping[label]
            if label in tax["non-supportive"]:
                y_new.append(1)
            elif label in tax["support"]:
                y_new.append(0)
            elif label in tax["non-supportive"]["no Label"]:
                y_new.append(0)
            elif label in tax["support"]["contingency"]:
                y_new.append(1)
            elif label in tax["support"]["evidence"]:
                y_new.append(2)
            else:
                print("Conversion Error!")

        return y_new


