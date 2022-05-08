import numpy as np
import load_datasets

import NaiveBayes # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
import util

#importer d'autres fichiers et classes si vous en avez développés
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import time

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Charger/lire les datasets
iris = load_datasets.load_iris_dataset(85)
wine = load_datasets.load_wine_dataset(85)
abalone = load_datasets.load_abalone_dataset(80)

datasets = [iris, wine, abalone]

print("============ KNN:\n")
# Initialisez vos paramètres
params = [1, 2, 3]
best_params = []

for data in datasets:
    best_err = 1.0
    for k in params: 
        knn = Knn.KNNClassifier(n_neighbors=k)
        cv_scores = util.cross_validation_scores(knn, data[0].astype(np.float64), data[1].astype(np.float64), 3)
        err = sum([(1 - scores['accuracy']) for scores in cv_scores]) / 3
        if best_err > err:
            best_err = err
            best_k = k
    best_params.append(best_k)


# Initialisez/instanciez vos classifieurs avec leurs paramètres
my_knn_clf = [Knn.KNNClassifier(n_neighbors=k) for k in best_params]
sklearn_knn_clf = [KNeighborsClassifier(n_neighbors=k) for k in best_params]

# Entrainez votre classifieur
for index, data in enumerate(datasets):
    my_knn_clf[index].train(data[0].astype(np.float64), data[1].astype(np.float64))

for index, data in enumerate(datasets):
    sklearn_knn_clf[index].fit(data[0].astype(np.float64), data[1].astype(np.float64))

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
# Tester votre classifieur
print("TRAIN:")

print("============ Custom: ")
for index, data in enumerate(datasets):
    t1 = time.time()
    result = my_knn_clf[index].evaluate(data[0].astype(np.float64), data[1].astype(np.float64))
    con_matrix = result.pop("con_matrix")
    print(con_matrix, "\n", result, "\n")
    t2 = time.time()
    print("Temps d'évaluation sur train KNN: %f" % (t2-t1), "data %d" % index)

print("============ Sklearn:")
for index, data in enumerate(datasets):
    scores = dict()
    y_pred = sklearn_knn_clf[index].predict(data[0].astype(np.float64))
    y_true = data[1].astype(np.float64)
    scores["accuracy"] = accuracy_score(y_true, y_pred)
    scores["precision"] = precision_score(y_true, y_pred, average="macro")
    scores["recall"] = recall_score(y_true, y_pred, average="macro")
    scores["f1_score"] = f1_score(y_true, y_pred, average="macro")
    print(multilabel_confusion_matrix(y_true, y_pred), "\n", scores, "\n")

"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
print("\n\nTEST:")

print("============ Custom: ")
for index, data in enumerate(datasets):
    t1 = time.time()
    result = my_knn_clf[index].evaluate(data[2].astype(np.float64), data[3].astype(np.float64))
    con_matrix = result.pop("con_matrix")
    print(con_matrix, "\n", result, "\n")
    t2 = time.time()
    print("Temps d'évaluation sur test KNN: %f" % (t2-t1), "data %d" % index)

print("============ Sklearn:")
for index, data in enumerate(datasets):
    scores = dict()
    y_pred = sklearn_knn_clf[index].predict(data[2].astype(np.float64))
    y_true = data[3].astype(np.float64)
    scores["accuracy"] = accuracy_score(y_true, y_pred)
    scores["precision"] = precision_score(y_true, y_pred, average="macro")
    scores["recall"] = recall_score(y_true, y_pred, average="macro")
    scores["f1_score"] = f1_score(y_true, y_pred, average="macro")
    print(multilabel_confusion_matrix(y_true, y_pred), "\n", scores, "\n")


# -----------------------------------------


print("============ CNB:\n")

# Initialisez/instanciez vos classifieurs avec leurs paramètres
naive_iris = NaiveBayes.NaiveBayes()
naive_wine = NaiveBayes.NaiveBayes()
naive_abalone = NaiveBayes.NaiveBayes()

skl_CNB_iris = GaussianNB()
skl_CNB_wine = GaussianNB()
skl_CNB_abalone = GaussianNB()

# Charger/lire les datasets
iris = load_datasets.load_iris_dataset(85)
wine = load_datasets.load_wine_dataset(85)
abalone = load_datasets.load_abalone_dataset(80)

iris_train = iris[0].astype(np.float64)
iris_train_labels = iris[1]
iris_test = iris[2].astype(np.float64)
iris_test_labels = iris[3]

wine_train = wine[0].astype(np.float64)
wine_train_labels = wine[1]
wine_test = wine[2].astype(np.float64)
wine_test_labels = wine[3]

abalone_train = abalone[0].astype(np.float64)
abalone_train_labels = abalone[1]
abalone_test = abalone[2].astype(np.float64)
abalone_test_labels = abalone[3]

# Entrainez votre classifieur
naive_iris.train(iris_train, iris_train_labels)
naive_wine.train(wine_train, wine_train_labels)
naive_abalone.train(abalone_train, abalone_train_labels)

skl_CNB_iris.fit(iris_train, iris_train_labels) 
skl_CNB_wine.fit(wine_train, wine_train_labels)
skl_CNB_abalone.fit(abalone_train, abalone_train_labels)


"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
metriques = ["Confusion Matrix", "Accuracy", "Precision", "Recall", "F1-score"]
#iris CNB
print("Évaluation de notre CNB sur iris d'entrainement.")
t1 = time.time()
iris_train_metriques = naive_iris.evaluate(iris_train, iris_train_labels)
for metr in metriques:
    print(metr, "\n", iris_train_metriques.get(metr))
t2 = time.time()
print("Temps d'évaluation sur train CNB: %f" % (t2-t1), "- data 1")

#wine CNB
print("Évaluation de notre CNB sur wine d'entrainement.")
t1 = time.time()
wine_train_metriques = naive_wine.evaluate(wine_train, wine_train_labels)
for metr in metriques:
    print(metr, "\n", wine_train_metriques.get(metr))
t2 = time.time()
print("Temps d'évaluation sur train CNB: %f" % (t2-t1), "- data 2")

#abalone CNB
print("Évaluation de notre CNB sur abalone d'entrainement.")
t1 = time.time()
abalone_train_metriques = naive_abalone.evaluate(abalone_train, abalone_train_labels)
for metr in metriques:
    print(metr, "\n", abalone_train_metriques.get(metr))
t2 = time.time()
print("Temps d'évaluation sur train CNB: %f" % (t2-t1), "- data 3")


# Tester votre classifieur

"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
#iris CNB
t1 = time.time()
print("Évaluation de notre CNB sur iris test.")
iris_test_metriques = naive_iris.evaluate(iris_test, iris_test_labels)
for metr in metriques:
    print(metr, "\n", iris_test_metriques.get(metr))
t2 = time.time()
print("Temps d'évaluation sur test CNB: %f" % (t2-t1), "- data 1")

#sklearn CNB on iris
print("Évaluation du CNB de sklearn sur iris test.")
pred_skl = skl_CNB_iris.predict(iris_test) 
skl_test_metric = util.all_metrics(pred_skl, iris_test_labels, skl_CNB_iris.classes_)
for metr in metriques:
    print(metr, "\n", skl_test_metric.get(metr))


#wine CNB
print("Évaluation de notre CNB sur wine test.")
t1 = time.time()
wine_test_metriques = naive_wine.evaluate(wine_test, wine_test_labels)
for metr in metriques:
    print(metr, "\n", wine_test_metriques.get(metr))
t2 = time.time()
print("Temps d'évaluation sur test CNB: %f" % (t2-t1), "- data 2")

#sklearn CNB on wine
print("Évaluation du CNB de sklearn sur wine test.")
pred_skl = skl_CNB_wine.predict(wine_test) 
skl_test_metric = util.all_metrics(pred_skl, wine_test_labels, skl_CNB_wine.classes_)
for metr in metriques:
    print(metr, "\n", skl_test_metric.get(metr))


#abalone CNB
t1 = time.time()
print("Évaluation de notre CNB sur abalone test.")
abalone_test_metriques = naive_abalone.evaluate(abalone_test, abalone_test_labels)
for metr in metriques:
    print(metr, "\n", abalone_test_metriques.get(metr))
t2 = time.time()
print("Temps d'évaluation sur test CNB: %f" % (t2-t1), "- data 3")

#sklearn CNB on abalone
print("Évaluation du CNB de sklearn sur abalone test.")
pred_skl = skl_CNB_abalone.predict(abalone_test) 
skl_test_metric = util.all_metrics(pred_skl, abalone_test_labels, skl_CNB_abalone.classes_)
for metr in metriques:
    print(metr, "\n", skl_test_metric.get(metr))
