import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score

#importing our dataset
cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
cars = pd.read_csv("car.data", header=None, names = cols)


#split into X and y
cars_X = cars.iloc[:,:-1]
cars_y = cars.iloc[:,-1]

# use one hot encoding for our x variables, all of which are categorical.
cars_X = pd.get_dummies(cars_X)
print(cars_X.head(), cars_y.head())




#splitting our data into test, training, and validation sets.
X_main, X_test, y_main, y_test = train_test_split(cars_X, cars_y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size= 0.2, random_state=0)

# Visualizing the categorizations of our y value (quality)
qual_count = cars["class"].value_counts()

fig, ax = plt.subplots()
ax.bar(qual_count.index, qual_count.values)
ax.plot()
plt.title("Counts per Car Quality")
plt.show()

# Prelimanry viewing of Models
models = [LogisticRegression(), RandomForestClassifier(), KNeighborsClassifier(), MLPClassifier(max_iter=200, solver="lbfgs")]
axes = [(0,0),(0,1),(1,0),(1,1)]
metrics = []
fig, ax = plt.subplots(2,2, figsize = (12,12))
print(list(zip(axes,models)))

for a, model in zip(axes, models):
    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    cm_disp = ConfusionMatrixDisplay(cm, display_labels= ["acc", "good", "unacc", "vgood"])
    cm_disp.plot(ax = ax[a[0],a[1]])
    cm_disp.ax_.set_title(model)
    #cm_disp.im_.colorbar.remove()
plt.show()



fig, ax = plt.subplots(len(models),1)
for i, model in enumerate(models):
    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    df = pd.DataFrame(report)
    df = df.round(2)
    ax[i].axis('off')
    ax[i].set_title(model)
    table = pd.plotting.table(ax[i],df,
                loc='center',
                cellLoc='center',
                colColours=['lightblue']*7
                )
plt.show()
    
    


depth = [50*i for i in range(1,11)]
width = [50*i for i in range(1,11)]
train_acc = [0]*len(depth)
val_acc = [0]*len(depth)

for i, d in enumerate(depth):
    NN_clf = MLPClassifier(max_iter=d, solver="lbfgs")
    NN_clf.fit(X_train, y_train)
    train_acc[i] = NN_clf.score(X_train, y_train)
    val_acc[i] = (NN_clf.score(X_val, y_val))

fig, ax = plt.subplots()
ax.plot(depth,train_acc)
ax.plot(depth,val_acc)
plt.title("Training vs Validation Accuracy per number of Iterations")
plt.xticks(depth)
plt.ylim(0.9,)
plt.show()

for w in width:
    NN_clf = MLPClassifier(max_iter=350, hidden_layer_sizes= w, solver="lbfgs")
    NN_clf.fit(X_train, y_train)
    train_acc[i] = NN_clf.score(X_train, y_train)
    val_acc[i] = NN_clf.score(X_val, y_val)


fig, ax = plt.subplots()
ax.plot(width,train_acc)
ax.plot(width,val_acc)
ax.legend()
plt.title("Training vs Validation Accuracy per number of\n Nodes per Hidden Layer (With a depth of 500)")
plt.xticks(width)
plt.ylim(0.9,)
plt.show()


fig, ax = plt.subplots()
final_clf = MLPClassifier(hidden_layer_sizes=200,max_iter=200, solver="lbfgs")
final_clf.fit(X_train,y_train)
y_pred = final_clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
cm_disp = ConfusionMatrixDisplay(cm, display_labels= ["acc", "good", "unacc", "vgood"])
cm_disp.plot(ax = ax)
cm_disp.ax_.set_title("Confusion Matrix for NN Classifier")
plt.show()
#cm_disp.im_.colorbar.remove()
df = pd.DataFrame(report)
df = df.round(2)
print(df)























