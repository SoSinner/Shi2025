import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

features_info = [
    ('age', 'числова'),
    ('workclass', 'категоріальна*'),
    ('fnlwgt', 'числова'),
    ('education', 'категоріальна*'),
    ('education-num', 'числова'),
    ('marital-status', 'категоріальна*'),
    ('occupation', 'категоріальна*'),
    ('relationship', 'категоріальна*'),
    ('race', 'категоріальна*'),
    ('sex', 'категоріальна*'),
    ('capital-gain', 'числова'),
    ('capital-loss', 'числова'),
    ('hours-per-week', 'числова'),
    ('native-country', 'категоріальна*')
]

input_file = 'income_data.txt'
max_datapoints = 25000
test_size = 0.2
random_state = 5

X, y = [], []
count_class1, count_class2 = 0, 0

with open(input_file, 'r') as f:
    for line in f:
        if '?' in line or line.strip() == '':
            continue  # Пропуск рядків з відсутніми даними
        data = line.strip().split(', ')
        if len(data) != 15:  # 14 ознак + мітка
            continue
        
        label = data[-1]
        features = data[:-1]
        
        # Балансування класів
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

X = np.array(X)
y = np.array(y)

label_encoders = []
for i in range(X.shape[1]):
    try:
        X[:, i] = X[:, i].astype(float).astype(int)
    except ValueError:
        # Кодування LabelEncoder для категоріальних ознак
        le = preprocessing.LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        label_encoders.append((i, le))  # Зберігаємо індекс та енкодер

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    stratify=y,
    random_state=random_state
)

# Навчання моделі SVM
classifier = OneVsOneClassifier(LinearSVC(
    random_state=random_state, 
    max_iter=10000, 
    dual='auto'
))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['<=50K', '>50K'], 
            yticklabels=['<=50K', '>50K'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

test_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 
            'Handlers-cleaners', 'Not-in-family', 'White', 'Male', 
            '0', '0', '40', 'United-States']

encoded_data = []
for i, value in enumerate(test_data):
    try:
        encoded_data.append(int(value))
    except ValueError:
        for (idx, le) in label_encoders:
            if idx == i:
                encoded_data.append(le.transform([value])[0])
                break

prediction = classifier.predict([encoded_data])
print(f"\nPredicted class: {'>50K' if prediction[0] == 1 else '<=50K'}")
