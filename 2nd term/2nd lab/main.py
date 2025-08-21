import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data     # входные данные
y = data.target   # выходные данные (классы)

print(f"Форма X: {X.shape}")

X_centered = X - np.mean(X, axis=0)

# 2. Реализация PCA
def pca_custom(X, K):
    # Шаг 1: Нормализация (стандартизация)
    X_meaned = X - np.mean(X, axis=0)

    # Шаг 2: Матрица ковариации
    cov_matrix = np.cov(X_meaned, rowvar=False)

    # Шаг 3: Собственные значения и векторы (оптимизированная версия)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

    # Шаг 4: Сортировка по убыванию собственных значений
    sorted_idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_idx]
    eigen_vectors = eigen_vectors[:, sorted_idx]

    # Шаг 5: Выбор K главных компонент
    eigen_vectors_k = eigen_vectors[:, :K]

    # Шаг 6: Проекция данных
    X_reduced = np.dot(X_meaned, eigen_vectors_k)

    # Расчет доли объясненной дисперсии
    explained_variance_ratio = eigen_values / np.sum(eigen_values)

    return X_reduced, explained_variance_ratio[:K]
    # return X_reduced, eigen_values

# 3. Визуализация 2D-проекции
# X_pca_custom, eigen_vals = pca_custom(X, 2)
X_pca_custom, custom_var_ratio = pca_custom(X_centered, 2)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_custom[:, 0], X_pca_custom[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.title("PCA (собственная реализация, 2 компоненты)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# 4. Реализация с помощью sklearn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
# X_pca_sklearn = pca.fit_transform(X_scaled)
X_pca_sklearn = pca.fit_transform(X_centered)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.title("PCA (Sklearn, 2 компоненты)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# 5. Сравнение результатов
print("\nДоля объяснённой дисперсии (sklearn):", pca.explained_variance_ratio_)
print("Суммарно объяснённая дисперсия:", np.sum(pca.explained_variance_ratio_))

print("\nСравнение объясненной дисперсии:")
print("Кастомная PCA:", custom_var_ratio)
print("Sklearn PCA:  ", pca.explained_variance_ratio_)
print("\nСуммарная дисперсия:")
print(f"Кастомная: {np.sum(custom_var_ratio):.4f}")
print(f"Sklearn:   {np.sum(pca.explained_variance_ratio_):.4f}")

# 6. Метод локтя — подбор оптимального K
# pca_full = PCA().fit(X_scaled)
pca_full = PCA().fit(X_centered)
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title("Метод локтя: выбор числа главных компонент")
plt.xlabel("Число компонент")
plt.ylabel("Накопленная объяснённая дисперсия")
plt.grid(True)
plt.axhline(y=0.95, color='red', linestyle='--', label='95% дисперсии')
plt.legend()
plt.show()
