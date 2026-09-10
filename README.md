# Clustering de Breast Cancer Wisconsin — K-means + PCA

Repositorio de la entrega: uso de framework de aprendizaje máquina
(scikit-learn) para la implementación de un algoritmo de clustering.

## Archivos

- **`kmeans_pca.py`**
  Implementación del pipeline de clustering usando scikit-learn: carga el
  dataset Breast Cancer Wisconsin (integrado en la librería, no requiere
  CSV externo), separa train/test, escala las features, reduce a 2
  componentes con PCA, entrena K-means, alinea los clusters con las
  etiquetas reales y muestra matriz de confusión y métricas en consola.
  Corre solo con:
  ```
  python kmeans_pca.py
  ```

- **`kmeans_breast_cancer_analysis.ipynb`**
  Notebook con el análisis completo: EDA (distribución de clases,
  distribución de features, correlaciones), justificación de PCA
  (varianza explicada), método del codo para justificar k=2,
  entrenamiento de K-means,
  visualización de clusters vs. clases reales, matriz de confusión y
  métricas.

- **`reporte_kmeans.docx`**
  Reporte con los resultados: dataset usado, metodología y configuración
  del framework, matriz de confusión y métricas justificadas, y
  análisis/conclusión del desempeño obtenido.

- **`README.md`**
  Este archivo.


