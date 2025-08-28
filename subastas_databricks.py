# Databricks notebook source
import pandas as pd

results_df = pd.read_excel('/Workspace/Users/leonardorubio1998@gmail.com/subasta_metrics.xlsx')
results_df

# COMMAND ----------

df = pd.read_csv('/Workspace/Users/leonardorubio1998@gmail.com/bids_nombre_parties_ruc.csv')

# COMMAND ----------

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import random

# Number of ocids to process (change this to your desired number)
random.seed(42)
num_ocids = 500 # Change to 20, 50, 100, etc.

# Filter ocids with >=4 bids and sample num_ocids
ocid_counts = df.groupby('ocid').size()
valid_ocids = ocid_counts[ocid_counts >= 5].index
if len(valid_ocids) < num_ocids:
    print(f"Warning: Only {len(valid_ocids)} ocids have >=4 bids. Using all available.")
    selected_ocids = valid_ocids
else:
    selected_ocids = random.sample(list(valid_ocids), num_ocids)

# Function to compute screens for a subgroup of 4 bids
def compute_screens(subgroup_bids):
    subgroup_bids = np.array(subgroup_bids)
    n = len(subgroup_bids)
    metrics = {}
    
    # CV
    std = np.std(subgroup_bids, ddof=1) if n > 1 else np.nan
    mean = np.mean(subgroup_bids) if n > 0 else np.nan
    metrics['CV'] = std / mean if mean != 0 else np.nan
    
    # Kurtosis
    metrics['KURTO'] = kurtosis(subgroup_bids, fisher=True, bias=False) if n >= 4 else np.nan
    
    # Spread
    max_b = np.max(subgroup_bids) if n > 0 else np.nan
    min_b = np.min(subgroup_bids) if n > 0 else np.nan
    metrics['SPD'] = (max_b - min_b) / min_b if min_b != 0 else np.nan
    
    # DIFFP, RD, RDNOR, RDALT
    sorted_b = np.sort(subgroup_bids)
    if n >= 2:
        b1 = sorted_b[0]
        b2 = sorted_b[1]
        d = b2 - b1
        metrics['DIFFP'] = (b2 - b1) / b1 if b1 != 0 else np.nan
        std_losing = np.std(sorted_b[1:], ddof=1) if n > 2 else np.nan
        metrics['RD'] = d / std_losing if std_losing != 0 and n > 2 else np.nan
        sum_adj = np.sum(np.diff(sorted_b))
        den = sum_adj / (n - 1) if n > 1 else np.nan
        metrics['RDNOR'] = d / den if den != 0 else np.nan
        if n >= 3:
            sum_adj_losing = np.sum(np.diff(sorted_b[1:]))
            den_alt = sum_adj_losing / (n - 2) if n > 2 else np.nan
            metrics['RDALT'] = d / den_alt if den_alt != 0 else np.nan
        else:
            metrics['RDALT'] = np.nan
    else:
        metrics['DIFFP'] = metrics['RD'] = metrics['RDNOR'] = metrics['RDALT'] = np.nan
    
    # Skewness
    metrics['SKEW'] = skew(subgroup_bids, bias=False) if n >= 3 else np.nan
    
    # KS
    if n >= 2:
        std_sub = np.std(subgroup_bids, ddof=1) if np.std(subgroup_bids, ddof=1) != 0 else 1
        std_bids = np.sort(subgroup_bids / std_sub)
        ranks = np.arange(1, n+1)
        d_plus = np.max(std_bids - ranks / (n + 1))
        d_minus = np.max(ranks / (n + 1) - std_bids)
        metrics['KS'] = max(d_plus, d_minus)
    else:
        metrics['KS'] = np.nan
    
    return metrics

# Compute summary screens for subgroups of 4 for selected ocids
summary_results = []
max_subgroups = 210  # Limit to 20 subgroups
for ocid in selected_ocids:
    bids = df[df['ocid'] == ocid]['amount'].dropna().values
    n_t = len(bids)
    if n_t < 4:  # Should not happen due to filter, but included for safety
        continue
    
    summary_metrics = {'ocid': ocid}
    
    # Subgroups of 4
    subgroups_4 = list(combinations(bids, 4))
    if len(subgroups_4) > max_subgroups:
        subgroups_4 = random.sample(subgroups_4, max_subgroups)
    if subgroups_4:
        screens_4 = [compute_screens(sub) for sub in subgroups_4]
        df_screens_4 = pd.DataFrame(screens_4)
        for screen in df_screens_4.columns:
            vals = df_screens_4[screen].replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                summary_metrics[f'MEAN4{screen}'] = vals.mean()
                summary_metrics[f'MEDIAN4{screen}'] = vals.median()
                summary_metrics[f'MIN4{screen}'] = vals.min()
                summary_metrics[f'MAX4{screen}'] = vals.max()
    
    summary_results.append(summary_metrics)

summary_df = pd.DataFrame(summary_results)

# Merge with tender-based screens
summary_df = summary_df.merge(results_df, on='ocid', how='left')

# Select features for clustering
feature_cols = [col for col in summary_df.columns if col.startswith(('MEAN4', 'MEDIAN4', 'MIN4', 'MAX4')) or col in ['cvt', 'kurtosis', 'spread', 'diffp', 'rdt', 'rdnor', 'skew', 'ks_statistic', 'mean', 'std_dev']]
X = summary_df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Impute missing values and scale
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# K-means clustering (2 clusters)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
summary_df['cluster'] = kmeans.fit_predict(X_scaled)

# Output ocid and cluster (0 or 1)
print(summary_df[['ocid', 'cluster']])


# Print cluster means for validation (ignoring inf)
cluster_means_clean = summary_df[['cvt', 'kurtosis', 'spread', 'diffp', 'rdt', 'rdnor', 'skew', 'ks_statistic', 'mean', 'std_dev', 'cluster']].replace([np.inf, -np.inf], np.nan)
print(cluster_means_clean.groupby('cluster')[['cvt', 'ks_statistic', 'rdt', 'diffp']].mean())

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. Calcular el número de ofertas por OCID y filtrar solo las con ≥4 ofertas
offers_count = df.groupby('ocid')['amount'].count().reset_index(name='num_offers')
offers_count = offers_count[offers_count['num_offers'] >= 4]  # Filtro clave

# 2. Combinar con la información de cluster
offers_count = offers_count.merge(summary_df[['ocid', 'cluster']], on='ocid', how='inner')

# 3. Configuración del gráfico
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

# 4. Paleta de colores mejorada
palette = {
    0: '#E63946',  # Rojo intenso - No competitivo
    1: '#457B9D',   # Azul profesional - Competitivo
    2: '#FFD166'    # Amarillo - Intermedio
}

# 5. Gráfico de densidad con inicio en 4
ax = sns.kdeplot(
    data=offers_count,
    x='num_offers',
    hue='cluster',
    palette=palette,
    linewidth=2.5,
    fill=True,
    alpha=0.15,
    common_norm=False,
    bw_adjust=0.5,
    clip=(4, None)
)

# 6. Personalización avanzada
plt.title('Distribución de Cantidad de Ofertas por OCID según Cluster', 
          fontsize=16, pad=20, weight='bold')
plt.xlabel('Número de Ofertas por Licitación (≥4)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)

# 7. Leyenda mejorada
legend_labels = [
    'Cluster 0', 
    'Cluster 1', 
    'Cluster 2'
]
ax.legend_.set_title('Cluster', prop={'size': 12})
for t, l in zip(ax.legend_.texts, legend_labels):
    t.set_text(l)
    t.set_fontsize(10)

# 8. Ajuste preciso de ejes
min_offers = 4
max_offers = 20  # Mostrar hasta el percentil 95
plt.xlim([min_offers, max_offers])

# 9. Líneas de referencia con anotaciones
for cluster, color in palette.items():
    median = offers_count[offers_count['cluster'] == cluster]['num_offers'].median()
    plt.axvline(median, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
    plt.text(
        median + 0.1, 
        plt.ylim()[1]*0.85 - cluster*0.05, 
        f'Mediana: {median:.0f}', 
        color=color,
        fontsize=10,
        weight='bold'
    )

# 10. Cuadrícula y formato
plt.grid(axis='y', alpha=0.3)
plt.xticks(range(min_offers, int(max_offers)+1))  # Marcas enteras en eje X

plt.tight_layout()
plt.show()

# 11. Estadísticas detalladas (solo para ≥4 ofertas)
print("\nEstadísticas de cantidad de ofertas (≥4) por cluster:")
stats = offers_count.groupby('cluster')['num_offers'].agg([
    ('OCIDs', 'count'),
    ('Media', 'mean'),
    ('Mediana', 'median'),
    ('Desv.Est.', 'std'),
    ('Mínimo', 'min'),
    ('Percentil 25', lambda x: x.quantile(0.25)),
    ('Percentil 75', lambda x: x.quantile(0.75)),
    ('Máximo', 'max')
])
print(stats.round(2))

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. Filtrar datos y unir con información de cluster
ocid_counts = df['ocid'].value_counts()
ocids_4plus = ocid_counts[ocid_counts >= 4].index
filtered_df = df[df['ocid'].isin(ocids_4plus)].merge(
    summary_df[['ocid', 'cluster']], 
    on='ocid', 
    how='inner'
)

# 2. Configuración del gráfico
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

# 3. Paleta de colores con nombres de clusters
palette = {
    0: ('#E63946', 'Cluster 0 '),
    1: ('#457B9D', 'Cluster 1'), 
    2: ('#FFD166', 'Cluster 2')
}

# 4. Gráfico de densidad
ax = sns.kdeplot(
    data=filtered_df,
    x='amount',
    hue='cluster',
    palette=[v[0] for v in palette.values()],  # Extraer códigos de color
    linewidth=2.5,
    fill=True,
    alpha=0.15,
    common_norm=False,
    bw_adjust=0.5,
    warn_singular=False
)

# 5. Personalización avanzada con números de cluster
plt.title('Distribución de Montos de Puja por Cluster (OCIDs con ≥4 ofertas)', 
          fontsize=16, pad=20, weight='bold')
plt.xlabel('Monto de la Puja', fontsize=12)
plt.ylabel('Densidad', fontsize=12)

# 6. Leyenda mejorada con números y descripciones
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=[palette[int(float(label))][1] for label in labels],  # Usar descripciones completas
    title='Clusters',
    fontsize=10,
    title_fontsize=12,
    frameon=True,
    framealpha=0.9
)

# 7. Ajustar ejes y formato
lower_bound = filtered_df['amount'].quantile(0.01)
upper_bound = filtered_df['amount'].quantile(0.95)
plt.xlim([lower_bound, upper_bound])
plt.gca().xaxis.set_major_formatter('${x:,.0f}')

# 8. Líneas de mediana
for cluster, (color, label) in palette.items():
    cluster_data = filtered_df[filtered_df['cluster'] == cluster]
    median = cluster_data['amount'].median()
    plt.axvline(median, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
    plt.text(
        median * 1.02,
        plt.ylim()[1] * (0.85 - cluster * 0.05),
        f'{label}\nMediana: ${median:,.0f}',
        color=color,
        fontsize=9,
        weight='bold',
        va='top'
    )

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 9. Estadísticas descriptivas con números de cluster
print("\nESTADÍSTICAS DE MONTOS POR CLUSTER (OCIDs con ≥4 ofertas)")
stats = filtered_df.groupby('cluster')['amount'].agg([
    ('N° OCIDs', lambda x: x.nunique()),
    ('N° Pujas', 'count'),
    ('Media', 'mean'),
    ('Mediana', 'median'),
    ('Desv.Est.', 'std'),
    ('Mínimo', 'min'),
    ('Percentil 25', lambda x: x.quantile(0.25)),
    ('Percentil 75', lambda x: x.quantile(0.75)),
    ('Máximo', 'max')
]).rename(index=palette)

# Formatear como moneda
def format_currency(x):
    if isinstance(x, (int, float)):
        return f"${x:,.2f}"
    return x

styled_stats = stats.style.format({
    'Media': format_currency,
    'Mediana': format_currency,
    'Desv.Est.': format_currency,
    'Mínimo': format_currency,
    'Percentil 25': format_currency,
    'Percentil 75': format_currency,
    'Máximo': format_currency
})

display(styled_stats.set_caption("Estadísticas Descriptivas por Cluster"))

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. Obtener la puja más pequeña por cada OCID
min_bids = df.groupby('ocid')['amount'].min().reset_index(name='min_amount')

# 2. Filtrar solo OCIDs con ≥4 ofertas y unir con cluster
ocid_counts = df['ocid'].value_counts()
ocids_4plus = ocid_counts[ocid_counts >= 4].index
filtered_min_bids = min_bids[min_bids['ocid'].isin(ocids_4plus)].merge(
    summary_df[['ocid', 'cluster']], 
    on='ocid', 
    how='inner'
)

# 3. Configuración del gráfico
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

# 4. Paleta de colores con identificación numérica de clusters
palette = {
    0: ('#E63946', 'Cluster 0 '),
    1: ('#457B9D', 'Cluster 1 '), 
    2: ('#FFD166', 'Cluster 2')
}

# 5. Gráfico de densidad de la puja mínima
ax = sns.kdeplot(
    data=filtered_min_bids,
    x='min_amount',
    hue='cluster',
    palette=[v[0] for v in palette.values()],
    linewidth=2.5,
    fill=True,
    alpha=0.15,
    common_norm=False,
    bw_adjust=0.6,
    warn_singular=False
)

# 6. Personalización del gráfico
plt.title('Distribución de las pujas ganadoras', 
          fontsize=16, pad=20, weight='bold')
plt.xlabel('Monto de la Puja Más Baja', fontsize=12)
plt.ylabel('Densidad', fontsize=12)

# 7. Leyenda con identificación numérica
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=[palette[int(float(label))][1] for label in labels],
    title='Clusters',
    fontsize=10,
    title_fontsize=12,
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

# 8. Ajuste de ejes y formato
q1 = filtered_min_bids['min_amount'].quantile(0.01)
q99 = filtered_min_bids['min_amount'].quantile(0.95)
plt.xlim([q1, q99])
plt.gca().xaxis.set_major_formatter('${x:,.0f}')

# 9. Líneas de mediana

for cluster, (color, label) in palette.items():
    cluster_data = filtered_min_bids[filtered_min_bids['cluster'] == cluster]['min_amount']
    
    # Mediana
    median = cluster_data.median()
    plt.axvline(median, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
    plt.text(
        median * 1.02, 
        plt.ylim()[1] * (0.85 - cluster * 0.05),
        f'{label}\nMediana: ${median:,.0f}',
        color=color,
        fontsize=9,
        weight='bold'
    )
    
    # Media
    mean = cluster_data.mean()
    plt.axvline(mean, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(
        mean * 1.02, 
        plt.ylim()[1] * (0.75 - cluster * 0.05),
        f'Media: ${mean:,.0f}',
        color=color,
        fontsize=9,
        weight='bold'
    )


plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()



# COMMAND ----------

from scipy.stats import mannwhitneyu

# Extraer los datos de min_amount por cluster
cluster0 = filtered_min_bids[filtered_min_bids['cluster']==0]['min_amount']
cluster1 = filtered_min_bids[filtered_min_bids['cluster']==1]['min_amount']
cluster2 = filtered_min_bids[filtered_min_bids['cluster']==2]['min_amount']


# COMMAND ----------

stat01, p01 = mannwhitneyu(cluster1, cluster0, alternative='less')
print(f"Cluster 1 < Cluster 0: U={stat01}, p-value={p01:.4f}")

stat12, p12 = mannwhitneyu(cluster1, cluster2, alternative='less')
print(f"Cluster 1 < Cluster 2: U={stat12}, p-value={p12:.4f}")



# COMMAND ----------

# Agrupar por clúster y calcular estadísticas
cluster_summary = summary_df.groupby('cluster').agg(
    num_ocids=('ocid', 'count'),
    mean_cvt=('cvt', 'mean'),
    mean_kurtosis=('kurtosis', 'mean'),
    mean_spread=('spread', 'mean'),
    mean_diffp=('diffp', 'mean'),
    mean_rdt=('rdt', 'mean'),
    mean_rdnor=('rdnor', 'mean'),
    mean_skew=('skew', 'mean'),
    mean_ks=('ks_statistic', 'mean')
).reset_index()

# Redondear para mejor presentación
cluster_summary = cluster_summary.round(3)

cluster_summary


# COMMAND ----------


# pip install ipywidgets
import ipywidgets as widgets
from IPython.display import display

cluster_selector = widgets.SelectMultiple(
    options=sorted(df_pca3['cluster'].unique()),
    value=tuple(sorted(df_pca3['cluster'].unique())),
    description='Clusters',
    disabled=False
)

def update_plot(change=None):
    sel = df_pca3[df_pca3['cluster'].isin(cluster_selector.value)]
    fig2 = px.scatter_3d(
        sel, x='PC1', y='PC2', z='PC3',
        color='cluster', hover_data=['ocid'],
        opacity=0.75, title='Filtro clusters'
    )
    fig2.show()

cluster_selector.observe(update_plot, names='value')
display(cluster_selector)
update_plot()

# COMMAND ----------

# Características usadas para clustering (por OCID)
feature_cols = [col for col in summary_df.columns if col.startswith(('MEAN4','MEDIAN4','MIN4','MAX4')) 
                or col in ['cvt','kurtosis','spread','diffp','rdt','rdnor','skew','ks_statistic','mean','std_dev']]
X_ocid = summary_df[feature_cols].replace([np.inf,-np.inf], np.nan)

# Imputar y escalar
imputer = SimpleImputer(strategy='median')
X_ocid_imputed = imputer.fit_transform(X_ocid)
scaler = StandardScaler()
X_ocid_scaled = scaler.fit_transform(X_ocid_imputed)

# KMeans sobre resumen OCID
kmeans_ocid = KMeans(n_clusters=3, random_state=42, n_init=10)
summary_df['cluster'] = kmeans_ocid.fit_predict(X_ocid_scaled)

# Ahora sí puedes calcular métricas
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
labels = summary_df['cluster'].values

sil_score = silhouette_score(X_ocid_scaled, labels)
ch_score = calinski_harabasz_score(X_ocid_scaled, labels)
db_score = davies_bouldin_score(X_ocid_scaled, labels)

print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Index:", ch_score)
print("Davies-Bouldin Index:", db_score)


# COMMAND ----------

from scipy.stats import f_oneway, kruskal

print("\nPruebas de Diferencias Estadísticas entre Clusters:")

features_to_test = ['cvt', 'ks_statistic', 'rdt', 'diffp']

for feature in features_to_test:
    # Obtener datos por cluster
    cluster_data = [summary_df[summary_df['cluster']==c][feature].dropna() for c in summary_df['cluster'].unique()]
    
    # ANOVA (diferencia de medias)
    f_stat, p_val = f_oneway(*cluster_data)
    
    # Kruskal-Wallis (no paramétrico)
    h_stat, h_pval = kruskal(*cluster_data)
    
    print(f"\nVariable {feature}:")
    print(f"  - ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
    print(f"  - Kruskal-Wallis: H={h_stat:.3f}, p={h_pval:.4f}")
