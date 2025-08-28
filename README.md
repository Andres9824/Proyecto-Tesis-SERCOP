# Propuesta para la detección de manipulación de ofertas en subastas inversas electrónicas dentro de las Compras Públicas Ecuatorianas con Aprendizaje No Supervisado

Este repositorio contiene el código y la documentación de un proyecto enfocado en la detección de posibles patrones de colusión en el sistema de compras públicas ecuatoriano (SERCOP) mediante la aplicación de técnicas de clustering.

[cite_start]El proyecto aborda la falta de un conjunto de datos etiquetado de manera confiable en el Ecuador, lo que justifica la implementación de un enfoque de aprendizaje no supervisado[cite: 9].

## Resumen del Proyecto

[cite_start]El objetivo principal es identificar grupos o clústeres de subastas inversas con características similares en sus patrones de oferta, lo que podría indicar comportamientos anticompetitivos o colusorios[cite: 8]. [cite_start]Para ello, se utilizan métricas estadísticas derivadas de la distribución de las ofertas [cite: 13] [cite_start]como "screens" o detectores[cite: 12], que han demostrado ser efectivos en la literatura académica.

## Tecnologías Utilizadas

* **Python**: Lenguaje de programación principal para el análisis de datos, la manipulación de datos (`Pandas`, `NumPy`) y la implementación de algoritmos de clustering (`Scikit-learn`).
* **Databricks**: Entorno de trabajo utilizado para el procesamiento distribuido de grandes volúmenes de datos, permitiendo una ejecución eficiente de los análisis.
* **SQL**: Para la consulta y gestión de la base de datos de origen de los datos del SERCOP.

## Metodología

1.  **Extracción de Datos**: Se extrajeron datos de subastas inversas del sistema de compras públicas.
2.  **Cálculo de Métricas**: Se computaron métricas estadísticas clave (`Coeficiente de Variación`, `Kurtosis`, `Diferencia Porcentual`, etc.) para cada proceso de subasta. [cite_start]Estas métricas actúan como las características del conjunto de datos[cite: 16].
3.  [cite_start]**Clustering**: Se aplicó un modelo de clustering [cite: 8] para agrupar las subastas en función de sus métricas. [cite_start]La evaluación del modelo se realizó utilizando métricas como el `Silhouette Score` y el `Davies-Bouldin Index` [cite: 23, 25] para confirmar la calidad de la agrupación.
4.  **Análisis de Clústeres**: Se analizó la distribución de las ofertas dentro de cada clúster para identificar anomalías. [cite_start]Se realizaron pruebas estadísticas (ANOVA, Kruskal-Wallis) para confirmar la diferencia significativa entre los grupos[cite: 58].
5.  [cite_start]**Interpretación**: Los hallazgos sugieren que ciertos clústeres presentan patrones de oferta que se asocian a una menor competencia y, potencialmente, a prácticas colusorias, en contraste con otros clústeres que muestran un comportamiento más competitivo[cite: 88, 89].

## Resultados Clave

* [cite_start]**Identificación de Clústeres**: Se identificaron clústeres de subastas con características de oferta distintas[cite: 18].
* [cite_start]**Patrones Anómalos**: Los análisis revelaron que ciertos clústeres, en particular el Clúster 0 y 2, tienen un `Coeficiente de Variación` bajo y una distribución de ofertas que se desvía de lo esperado en una subasta competitiva[cite: 88].
* [cite_start]**Impacto Económico**: Se demostró que las pujas ganadoras en clústeres considerados competitivos (`Clúster 1`) tienden a ser significativamente menores que las de los clústeres con indicios de colusión, lo que subraya el perjuicio al Estado[cite: 124, 134].
