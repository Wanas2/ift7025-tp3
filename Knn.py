import numpy as np
from util import *


class KNNClassifier:

    def __init__(self, n_neighbors=3, **kwargs):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.n_neighbors = n_neighbors
        
    def train(self, train, train_labels):
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)
        
        train_labels : est une matrice numpy de taille nx1
        
        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        
        """
        self.features_ = train
        self.labels_ = train_labels

    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        nearest_labels = list()
        for neighbor in self._get_nearest_neighbors(self.features_, x):
            nearest_labels.append(neighbor[-1])

        return max(set(nearest_labels), key=nearest_labels.count)
        
        
    def evaluate(self, X, y):
        """
        c'est la méthode qui va evaluer votre modèle sur les données X
        l'argument X est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)
        
        y : est une matrice numpy de taille nx1
        
        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        """
        predictions = list()
        for x in X:
            predictions.append(self.predict(x))

        predictions = np.array(predictions)

        unique_labels = set(self.labels_)
        unique_labels.update(y)
        self.unique_labels_ = list(unique_labels)

        self.n_labels_ = len(unique_labels)
        
        return self._get_metrics(predictions, y)


    # Vous pouvez rajouter d'autres méthodes et fonctions,
    # il suffit juste de les commenter.
    def _get_nearest_neighbors(self, X, element):
        distances = list()
        
        for i, x in enumerate(X):
            dist = euclidian_distance(element, x)
            distances.append((np.append(x, [self.labels_[i]]), dist))
        
        distances.sort(key=lambda t: t[1])

        neighbors = list()
        for i in range(self.n_neighbors):
            neighbors.append(distances[i][0])
        
        return neighbors

    def _get_metrics(self, y_pred, y_true):
        if self.n_labels_ == 2:
            con_matrix = binary_confusion_matrix(y_pred, y_true, self.unique_labels_[0])
            accuracy = accuracy_metrics(con_matrix)
            precision = precision_metrics(con_matrix)
            recall = recall_metrics(con_matrix)
            f1_score = f1_score_metrics(con_matrix)
        elif self.n_labels_ > 2:
            con_matrix = multilabel_confusion_matrix(y_pred, y_true, self.unique_labels_)
            
            accuracy = sum([accuracy_metrics(con_matrix[i]) for i in range(self.n_labels_)]) / self.n_labels_  

            precision = sum([precision_metrics(con_matrix[i]) for i in range(self.n_labels_)]) / self.n_labels_   

            recall = sum([recall_metrics(con_matrix[i]) for i in range(self.n_labels_)]) / self.n_labels_

            f1_score = sum([f1_score_metrics(con_matrix[i]) for i in range(self.n_labels_)]) / self.n_labels_

        return ({
            "con_matrix": con_matrix,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })
