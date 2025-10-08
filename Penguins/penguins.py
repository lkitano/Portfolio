import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report, confusion_matrix
#from mlxtend.plotting import plot_decision_regions 


penguins = pd.read_csv("penguins.csv")
penguins = penguins[["species","bill_length_mm","bill_depth_mm"]]
species = ["Adelie", "Gentoo", "Chinstrap"]
penguins = penguins.dropna()

print(penguins.head())





penguins_X = penguins.iloc[:,1:].values
penguins_y = penguins.iloc[:,0].values


penguins_y_int = [0]*len(penguins_y)
for i, p in enumerate(penguins_y):
    if p == "Adelie":
        penguins_y_int[i] = 0
    elif p == "Chinstrap":
        penguins_y_int[i] = 1
    else:
        penguins_y_int[i] = 2



#print(penguins_X)
#print(penguins_y)


X_train,X_test, y_train, y_test = train_test_split(penguins_X, penguins_y, test_size = 0.2)





'''
for i, k in enumerate(k_neighbors):
    print(i,k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    test_acc[i] = knn.score(X_test, y_test)
    train_acc[i] = knn.score(X_train,y_train)



plt.title("kNN Classifier: Accuracy per number of Neighbors")
plt.plot(k_neighbors, test_acc)
plt.plot(k_neighbors, train_acc)
plt.xticks(k_neighbors)
plt.show()
'''

#lr = LogisticRegression().fit(scaled_X_train,y_train)
log_clf = knn_clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("logr", LogisticRegression())]
)

log_clf.fit(X_train, y_train)


fig, (ax1, ax2) = plt.subplots(2)
log_disp = DecisionBoundaryDisplay.from_estimator(
    log_clf, X_test, response_method="predict",xlabel="bill length (mm)", ylabel="bill depth (mm)",
    alpha = 0.5, ax = ax1
)

scatter = log_disp.ax_.scatter(penguins_X[:, 0], penguins_X[:, 1], c = penguins_y_int, edgecolors="black")
#print("here!!!!!!", scatter.legend_elements()[0])
log_disp.ax_.legend(
    handles = scatter.legend_elements()[0],
    labels= species,
    loc="upper left",
    title="Species",
    )
log_disp.plot()

y_pred = log_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels = species, ax = ax2)
cm_display.plot()





plt.show()



print("Logistic regression classifier score: ", log_clf.score(X_test, y_test))











k_neighbors = [3,5,7,9,11,13]
test_acc = [0]*len(k_neighbors)
train_acc = [0]*len(k_neighbors)


#fig, ax = plt.subplots(2,3)

for i, k in enumerate(k_neighbors):

    knn_clf = Pipeline(
        steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors = k))]
        )
    knn_clf.fit(X_train,y_train)
    test_acc[i] = knn_clf.score(X_test, y_test)
    train_acc[i] = knn_clf.score(X_train,y_train)
    knn_disp = DecisionBoundaryDisplay.from_estimator(
        knn_clf, X_test, response_method="predict",xlabel="bill length (mm)", ylabel="body mass (g)",
        alpha = 0.5, 
        )
    scatter2 = knn_disp.ax_.scatter(penguins_X[:, 0], penguins_X[:, 1], c = penguins_y_int, edgecolors="black")
    knn_disp.ax_.legend(
        handles = scatter2.legend_elements()[0],
        labels= species,
        loc="upper left",
        title="Species",
    )
    knn_disp.ax_.set_title(
        f"Decision Boundaries for k = {k} neighbors"
    )

    #plt.show()
