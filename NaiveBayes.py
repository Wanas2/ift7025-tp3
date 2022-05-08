"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
    * train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
    * predict 	: pour prédire la classe d'un exemple donné.
    * evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
import util

# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class NaiveBayes: #nom de la class à changer

    def __init__(self, **kwargs):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        
        
        
    def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)
        
        train_labels : est une matrice numpy de taille nx1
        
        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        
        """
        self.n_samples, self.n_features = train.shape
        self.classes = np.unique(train_labels)
        self.n_classes = len(self.classes)
        self.means = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self.std = np.zeros((self.n_classes, self.n_features), dtype=np.float64)

        #Calcul des probabilités a priori P(Y=y)
        self.prior = []
        for classe in self.classes:
            self.prior.append(len(train_labels[train_labels==classe])/self.n_samples)

        #Calcul des moyennes et écart-type des distributions de 
        #probabilités conditionnelle P(X=x|Y=y) sur tout le jeu de donnée
        for classe_idx, classe in enumerate(self.classes):
            cond = train[train_labels==classe]
            for feat_idx in range(self.n_features):
                self.means[classe_idx,feat_idx] = cond[:,feat_idx].mean()
                self.std[classe_idx,feat_idx] = cond[:,feat_idx].std()

        
    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        post_prob = np.zeros(self.n_classes, dtype=np.float64)
        for classe_idx, classe in enumerate(self.classes):
            posterior = np.sum(np.log(self._gaussianProb(x, self.means[classe_idx], self.std[classe_idx])))
            post_prob[classe_idx] = np.log(self.prior[classe_idx]) + posterior

        return self.classes[np.argmax(post_prob)]
        
        
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
        predictions = np.zeros(len(X), dtype='<U20')
        for idx, x in enumerate(X):
            predictions[idx] = (self.predict(x))
        
        return self._metrics(predictions, y)
        
    
    # Vous pouvez rajouter d'autres méthodes et fonctions,
    # il suffit juste de les commenter
    
    def _gaussianProb(self, x, mean, std):
        """
        Calcul de la densité de la loi Gaussienne
        """
        prob = np.exp(-(x-mean)**2 / (2 * std**2)) / (np.sqrt(2*np.pi) * std)
        for i in range(len(prob)):
            if prob[i] == 0:
                prob[i] = 0.00001
        return prob

    
    def _metrics(self, y_pred, y_true):
        weight = []
        for classe in np.unique(y_true):
            weight.append(len(y_true[y_true==classe])/len(y_true))
        #Une seule matrice de confusion si on a un problème de classification binaire
        if self.n_classes == 2:
            con_matrix = util.binary_confusion_matrix(y_pred, y_true, self.classes[1])
            accuracy = util.accuracy_metrics(con_matrix)
            precision = util.precision_metrics(con_matrix)
            recall = util.recall_metrics(con_matrix)
            f1_score = util.f1_score_metrics(con_matrix)

            return {"Confusion Matrix":con_matrix, "Accuracy":accuracy, "Precision":precision, "Recall":recall, "F1-score":f1_score}
        #Approche un-contre-tous si on n'a pas un problème de classification binaire
        elif self.n_classes > 2:
            con_matrix = util.multilabel_confusion_matrix(y_pred, y_true, self.classes)
            accuracy = np.zeros(self.n_classes, dtype=np.float64)
            precision = np.zeros(self.n_classes, dtype=np.float64)
            recall = np.zeros(self.n_classes, dtype=np.float64)
            f1_score = np.zeros(self.n_classes, dtype=np.float64)

            for idx_classe, classe in enumerate(self.classes):
                accuracy[idx_classe] = util.accuracy_metrics(con_matrix[idx_classe])
                precision[idx_classe] = util.precision_metrics(con_matrix[idx_classe])
                recall[idx_classe] = util.recall_metrics(con_matrix[idx_classe])
                f1_score[idx_classe] = util.f1_score_metrics(con_matrix[idx_classe])
            accuracy_mean = np.average(accuracy, weights=weight)
            precision_mean = np.average(precision, weights=weight)
            recall_mean = np.average(recall, weights=weight)
            f1_score_mean = np.average(f1_score, weights=weight)
            return {"Confusion Matrix":con_matrix, "Accuracy":accuracy_mean, "Precision":precision_mean, "Recall":recall_mean, "F1-score":f1_score_mean}
        