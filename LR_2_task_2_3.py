import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

iris = load_iris()
dataset = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['class']
)
dataset['class'] = dataset['class'].map({i: name for i, name in enumerate(iris.target_names)})

print("=== Інформація про датасет ===")
print("Ключі:", list(iris.keys()))
print("\nОпис датасету:\n", iris['DESCR'][:193] + "...")
print("\nНазви класів:", iris['target_names'])
print("Назви ознак:", iris['feature_names'])
print("\nПерші 5 записів:\n", dataset.head())
print("\nРозподіл за класами:\n", dataset['class'].value_counts())

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.suptitle("Діаграми розмаху ознак")
plt.tight_layout()

plt.subplot(2, 2, 2)
dataset.hist()
plt.suptitle("Гістограми розподілу ознак")
plt.tight_layout()

plt.subplot(2, 1, 2)
pd.plotting.scatter_matrix(dataset.iloc[:, :4], figsize=(12, 10), 
                          c=iris.target, marker='o', alpha=0.8)
plt.suptitle("Матриця діаграм розсіювання")
plt.tight_layout()
plt.show()

X = dataset.iloc[:, :4].values
y = dataset['class'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

models = [
    ('LR', LogisticRegression(max_iter=200)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

results = []
print("\n=== Результати 10-кратної крос-валідації ===")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    print(f"{name}: {cv_results.mean():.3f} ± {cv_results.std():.3f}")

plt.figure(figsize=(10,6))
plt.boxplot(results, labels=[name for name, _ in models])
plt.title('Порівняння точності алгоритмів')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1.05)
plt.grid(True)
plt.show()

best_model = SVC(gamma='auto')
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_val)

print("\n=== Оцінка на валідаційному наборі ===")
print(f"Точність: {accuracy_score(y_val, predictions):.3f}")
print("\nМатриця плутанини:")
cm = confusion_matrix(y_val, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Прогноз')
plt.ylabel('Факт')
plt.title('Матриця плутанини (SVM)')
plt.show()

print("\nЗвіт класифікації:\n", classification_report(y_val, predictions))

new_sample = [[5.0, 2.9, 1.0, 0.2]]
predicted_class = best_model.predict(new_sample)
print(f"\nПрогноз для нового зразка {new_sample[0]}: {predicted_class[0]}")
