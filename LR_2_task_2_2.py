import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

features_info = [
    ('age', 'числова'),
    ('workclass', 'категоріальна'),
    ('fnlwgt', 'числова'),
    ('education', 'категоріальна'),
    ('education-num', 'числова'),
    ('marital-status', 'категоріальна'),
    ('occupation', 'категоріальна'),
    ('relationship', 'категоріальна'),
    ('race', 'категоріальна'),
    ('sex', 'категоріальна'),
    ('capital-gain', 'числова'),
    ('capital-loss', 'числова'),
    ('hours-per-week', 'числова'),
    ('native-country', 'категоріальна')
]

input_file = 'income_data.txt'
max_datapoints = 25000
test_size = 0.2
random_state = 5
poly_degree = 8 
class_labels = ['<=50K', '>50K']

X, y = [], []
count_class1, count_class2 = 0, 0

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
            
            if label == '<=50K' and count_class1 < max_datapoints:
                X.append(features)
                y.append(0)
                count_class1 += 1
            elif label == '>50K' and count_class2 < max_datapoints:
                X.append(features)
                y.append(1)
                count_class2 += 1
            
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break
except FileNotFoundError:
    print(f"Файл {input_file} не знайдено!")
    exit()

X = np.array(X)
y = np.array(y)

label_encoders = []
for col_idx in range(X.shape[1]):
    try:
        X[:, col_idx] = X[:, col_idx].astype(float).astype(int)
    except ValueErro:
        le = preprocessing.LabelEncoder()
        X[:, col_idx] = le.fit_transform(X[:, col_idx])
        label_encoders.append((col_idx, le))  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    stratify=y,
    random_state=random_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_svm(kernel_type, params, X_tr, y_tr, X_te, y_te):
    print(f"\n=== Модель з ядром {kernel_type.upper()} ===")
    
    model = SVC(kernel=kernel_type, random_state=random_state, **params)
    
    start_time = time.time()
    model.fit(X_tr, y_tr)
    print(f"Час навчання: {time.time() - start_time:.2f} сек")
    
    y_pred = model.predict(X_te)
    
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=class_labels, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_te, y_pred):.3f}")
    
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Матриця плутанини ({kernel_type.upper()})')
    plt.xlabel('Прогноз')
    plt.ylabel('Факт')
    plt.show()

evaluate_svm('poly', {'degree': poly_degree, 'gamma': 'auto'}, 
             X_train_scaled, y_train, X_test_scaled, y_test)

evaluate_svm('rbf', {'gamma': 'scale'}, 
             X_train_scaled, y_train, X_test_scaled, y_test)

evaluate_svm('sigmoid', {'gamma': 'scale'}, 
             X_train_scaled, y_train, X_test_scaled, y_test)
