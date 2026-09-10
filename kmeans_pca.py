# Los metodos de aprendizaje no supervisado no tienen matriz de confusion, accuracy, etc.
# porque no se tiene una etiqueta (no hay punto de comparacion), a diferencia del supervisado.

# Para este ejercicio se tomo un dataset con etiquetas, pero no se toman en cuenta para el clustering.
# Se aplica PCA y K-means para ver como K-means forma los clusters,
# y despues se compara contra las etiquetas reales, con el objetivo de ver que tanto
# la agrupacion de K-means coincide con las clases reales del ejemplo.


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


N_CLUSTERS = 2
RANDOM_STATE = 42


def cargar_datos():
    """Carga Breast Cancer Wisconsin"""
    datos = load_breast_cancer()
    X = datos.data
    y = datos.target  # 0 = malignant, 1 = benign
    nombres_clase = datos.target_names
    return X, y, nombres_clase


def alinear_clusters_con_etiquetas(clusters, y_real, n_clusters):
    """
    Este mapeo se calcula SOLO con el conjunto de entrenamiento.
    """
    mapeo = {}
    for cluster_id in range(n_clusters):
        etiquetas_en_cluster = y_real[clusters == cluster_id]
        if len(etiquetas_en_cluster) == 0:
            mapeo[cluster_id] = 0
            continue
        valores, conteos = np.unique(etiquetas_en_cluster, return_counts=True)
        mapeo[cluster_id] = valores[np.argmax(conteos)]
    return mapeo


def aplicar_mapeo(clusters, mapeo):
    """Traduce numeros de cluster a etiquetas reales usando el mapeo dado."""
    return np.array([mapeo[c] for c in clusters])


def imprimir_evaluacion(nombre_conjunto, y_real, y_pred, nombres_clase):
    print(f"\n--- {nombre_conjunto} ---")

    matriz = confusion_matrix(y_real, y_pred)
    print("Matriz de confusion:")
    print(f"                  Pred: {nombres_clase[0]:<10} Pred: {nombres_clase[1]:<10}")
    print(f"Real: {nombres_clase[0]:<10}   {matriz[0][0]:>10}       {matriz[0][1]:>10}")
    print(f"Real: {nombres_clase[1]:<10}   {matriz[1][0]:>10}       {matriz[1][1]:>10}")

    print("\nMetricas:")
    print(f"  accuracy : {accuracy_score(y_real, y_pred):.3f}")
    print(f"  precision: {precision_score(y_real, y_pred):.3f}")
    print(f"  recall   : {recall_score(y_real, y_pred):.3f}")
    print(f"  f1       : {f1_score(y_real, y_pred):.3f}")


def graficar_pca(X_pca, etiquetas, titulo, ruta_salida):
    plt.figure(figsize=(6, 5))
    dispersion = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=etiquetas, cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title(titulo)
    plt.colorbar(dispersion, label="Clase")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main():
    X, y, nombres_clase = cargar_datos()
    print(f"Dataset: Breast Cancer Wisconsin  ->  {X.shape[0]} filas, {X.shape[1]} features")
    print(f"Clases: {nombres_clase[0]} (0), {nombres_clase[1]} (1)")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} filas   Test: {X_test.shape[0]} filas")

    # escalado para aplicar PCA y K means
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    # PCA: 30 features -> 2 componentes
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_esc)
    X_test_pca = pca.transform(X_test_esc)
    varianza_explicada = pca.explained_variance_ratio_
    print(f"\nVarianza explicada por las 2 componentes: "
          f"{varianza_explicada[0]:.1%} + {varianza_explicada[1]:.1%} "
          f"= {sum(varianza_explicada):.1%}")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    clusters_train = kmeans.fit_predict(X_train_pca)
    clusters_test = kmeans.predict(X_test_pca)

    # alinear clusters con las etiquetas reales (solo con train)
    mapeo = alinear_clusters_con_etiquetas(clusters_train, y_train, N_CLUSTERS)
    pred_train = aplicar_mapeo(clusters_train, mapeo)
    pred_test = aplicar_mapeo(clusters_test, mapeo)

    # evaluacion
    imprimir_evaluacion("Entrenamiento", y_train, pred_train, nombres_clase)
    imprimir_evaluacion("Prueba", y_test, pred_test, nombres_clase)

    # graficas para el reporte
    graficar_pca(X_test_pca, y_test, "Clases reales (test)", "pca_reales.png")
    graficar_pca(X_test_pca, pred_test, "Clusters de K-means (test)", "pca_clusters.png")
    print("\nGraficas guardadas: pca_reales.png, pca_clusters.png")


if __name__ == "__main__":
    main()
