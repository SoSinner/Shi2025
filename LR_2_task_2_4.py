import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

input_file = 'income_data.txt'
max_datapoints = 25000
test_size = 0.2
random_state = 42
class_labels = ['<=50K', '>50K']
feature_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

def load_and_preprocess_data():
    """Завантаження та попередня обробка даних"""
    X, y = [], []
    counts = {0: 0, 1: 0}  
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if '?' in line or not line.strip():
                    continue
                
                data = line.strip().split(', ')
                if len(data) != 15: 
                    continue
                
                label = data[-1]
                features = data[:-1]

                if label == '<=50K' and counts[0] < max_datapoints:
                    X.append(features)
                    y.append(0)
                    counts[0] += 1
                elif label == '>50K' and counts[1] < max_datapoints:
                    X.append(features)
                    y.append(1)
                    counts[1] += 1
                
                if all(v >= max_datapoints for v in counts.values()):
                    break
        
        return np.array(X), np.array(y)
    
    except FileNotFoundError:
        print(f"Файл {input_file} не знайдено!")
        exit()

X, y = load_and_preprocess_data()

def encode_features(X_data):
    """Кодування категоріальних ознак"""
    label_encoders = []
    X_encoded = np.empty(X_data.shape, dtype=int)
    
    for col_idx in range(X_data.shape[1]):
        try:
            X_encoded[:, col_idx] = X_data[:, col_idx].astype(float).astype(int)
        except ValueError:
            le = preprocessing.LabelEncoder()
            X_encoded[:, col_idx] = le.fit_transform(X_data[:, col_idx])
            label_encoders.append((col_idx, le))
    
    return X_encoded, label_encoders

X_encoded, encoders = encode_features(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=test_size,
    stratify=y,
    random_state=random_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def compare_classifiers(models, X, y):
    """Порівняння алгоритмів з використанням стратифікованої крос-валідації"""
    results = []
    print("\n=== Результати 10-кратної крос-валідації ===")
    
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        results.append(cv_scores)
        print(f"{name}: Середнє = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    plt.figure(figsize=(12, 6))
    plt.boxplot(results, labels=[name for name, _ in models])
    plt.title('Порівняння точності алгоритмів (10-fold CV)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    return results

models = [
    ('LR', LogisticRegression(solver='liblinear', max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=random_state)),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto', random_state=random_state))
]

cv_results = compare_classifiers(models, X_train_scaled, y_train)

def evaluate_best_model(model, X_train, y_train, X_test, y_test):
    """Оцінка найкращої моделі на тестовому наборі"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n=== Результати на тестовому наборі ===")
    print(f"Точність: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Прогноз')
    plt.ylabel('Факт')
    plt.title('Матриця плутанини (Тестовий набір)')
    plt.show()

best_model = SVC(gamma='auto', random_state=random_state)
evaluate_best_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test)

print("\n--- Завдання 2.4 Завершено ---")
