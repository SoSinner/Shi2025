import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, matthews_corrcoef,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,
    stratify=y,
    random_state=42
)

# - tol=1e-2: Точність для зупинки ітерацій
# - solver='sag': Використання Stochastic Average Gradient 
clf = RidgeClassifier(tol=1e-2, solver='sag')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("=== Основні метрики якості ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"Matthews Corrcoef: {matthews_corrcoef(y_test, y_pred):.4f}\n")

print("=== Звіт класифікації ===")
print(classification_report(y_test, y_pred, target_names=class_names))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Матриця плутанини для RidgeClassifier', pad=20)
plt.xlabel('Передбачені класи')
plt.ylabel('Справжні класи')
plt.tight_layout()

try:
    plt.savefig("Confusion.jpg", dpi=300)
    print("Матрицю плутанини збережено як 'Confusion.jpg'")
except Exception as e:
    print(f"Помилка збереження: {str(e)}")

plt.show()

print("\n=== Пояснення метрик ===")
print("1. Коефіцієнт Коена Каппа (Cohen's Kappa):")
print("   - Вимірює узгодженість між передбаченнями та реальними класами")
print("   - Діапазон: від -1 (повна невідповідність) до 1 (ідеальна відповідність)\n")

print("2. Коефіцієнт кореляції Метьюза (MCC):")
print("   - Багатокласова узагальнена міра кореляції між прогнозами та реальними значеннями")
print("   - Діапазон: від -1 до 1, де 1 - ідеальний прогноз\n")

print("=== Налаштування RidgeClassifier ===")
print("- tol=1e-2: Допустима похибка для зупинки ітераційного алгоритму")
print("- solver='sag': Використання Stochastic Average Gradient для великих датасетів")
print("- alpha=1.0 (default): Параметр регуляризації (збільшення зменшує перетренування)")
