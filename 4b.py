
import pandas as pd

seeds_df = pd.read_csv('seeds-less-rows.csv')
varieties = list(seeds_df.pop('grain_variety'))
samples = seeds_df.values

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

mergings = linkage(samples, method='complete')

dendrogram(mergings,
           labels=varieties,
           leaf_rotation=60,
           leaf_font_size=12,
)
plt.show()

from scipy.cluster.hierarchy import fcluster

labels = fcluster(mergings, 6, criterion='distance')

df = pd.DataFrame({'labels': labels, 'varieties': varieties})

ct = pd.crosstab(df['labels'], df['varieties'])

ct
