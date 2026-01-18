# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report





# dataset = pd.read_csv("Attrition.csv")


# print(dataset.head())
# print(dataset.shape)
# print(dataset.columns)







# print(dataset.describe())
# print(dataset.groupby("Attrition").size())








# label_encoder = LabelEncoder()

# for column in dataset.columns:
#     if dataset[column].dtype == 'object':
#         dataset[column] = label_encoder.fit_transform(dataset[column])







# dataset.plot(kind='box', subplots=True, layout=(7,5), figsize=(20,18), sharex=False, sharey=False)
# plt.tight_layout()
# plt.show()

# exit()




# array = dataset.values

# X = array[:, 0:-1]   # all columns except last
# Y = array[:, -1]    # Attrition








# X_train, X_validation, Y_train, Y_validation = train_test_split(
#     X, Y, test_size=0.2, random_state=1
# )






# models = []
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('LR', LogisticRegression(max_iter=1000)))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))








# results = []
# names = []

# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print(f"{name}: Mean={cv_results.mean():.4f}, Std={cv_results.std():.4f}")








# plt.boxplot(results, labels=names)
# plt.title("Algorithm Comparison")
# plt.show()






# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_validation = scaler.transform(X_validation)








# final_model = SVC(kernel='rbf', gamma='scale', C=1.0)
# final_model.fit(X_train, Y_train)

# predictions = final_model.predict(X_validation)







# print("Accuracy:", accuracy_score(Y_validation, predictions))
# print("Confusion Matrix:\n", confusion_matrix(Y_validation, predictions))
# print("Classification Report:\n", classification_report(Y_validation, predictions))






# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report







# from sklearn.preprocessing import LabelEncoder

# label_encoder = LabelEncoder()

# for column in dataset.columns:
#     if dataset[column].dtype == 'object':
#         dataset[column] = label_encoder.fit_transform(dataset[column])








# X = dataset.drop("Attrition", axis=1)
# y = dataset["Attrition"]








# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )








# models = []
# models.append(("Logistic Regression", LogisticRegression(max_iter=1000)))
# models.append(("KNN", KNeighborsClassifier()))
# models.append(("Decision Tree", DecisionTreeClassifier()))
# models.append(("Naive Bayes", GaussianNB()))
# models.append(("SVM", SVC()))










# print("Model Performance (Cross-Validation Accuracy):\n")

# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     print(f"{name}: Mean Accuracy = {cv_results.mean():.4f}")











# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)








# final_model = SVC(kernel="rbf", C=1.0, gamma="scale")
# final_model.fit(X_train_scaled, y_train)










# y_pred = final_model.predict(X_test_scaled)

# print("\nFinal Model Evaluation:\n")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))













# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# for col in dataset.columns:
#     if dataset[col].dtype == 'object':
#         dataset[col] = LabelEncoder().fit_transform(dataset[col])

# X = dataset.drop("Attrition", axis=1)
# y = dataset["Attrition"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))



# print("Accuracy:", model.score(X_test, y_test)))






# print("\n--- MODEL OUTPUT ---\n")



import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



dataset = pd.read_csv("Attrition.csv")
print("Dataset loaded")



le = LabelEncoder()

for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = le.fit_transform(dataset[col])




X = dataset.drop("Attrition", axis=1)
y = dataset["Attrition"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)




y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))






from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name} Accuracy: {acc:.4f}")





print(dataset.shape)
print(dataset['Attrition'].value_counts())
print(dataset.describe())




dataset['Attrition'].value_counts().plot(kind='bar', title='Attrition Distribution')
plt.show()
