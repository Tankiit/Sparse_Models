import matplotlib.pyplot as plt
import numpy as np

# Sample data
datasets = ['CUB', 'SUN', 'ImageNet']
models = ['Label-Free CBM', 'CDM', 'SparseCBM']

# Mock data for Concept Consistency and Retrieval Performance
consistency_scores = {
    'CUB': [0.8, 0.85, 0.9],
    'SUN': [0.7, 0.75, 0.78],
    'ImageNet': [0.65, 0.7, 0.8]
}

retrieval_scores = {
    'CUB': [0.6, 0.65, 0.7],
    'SUN': [0.55, 0.6, 0.65],
    'ImageNet': [0.5, 0.58, 0.68]
}

# Number of datasets
n_datasets = len(datasets)
index = np.arange(n_datasets)  # the x locations for the groups
width = 0.25  # the width of the bars

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart for Concept Consistency
for i, model in enumerate(models):
    consistency = [consistency_scores[dataset][i] for dataset in datasets]
    ax1.bar(index + i*width, consistency, width, label=model)

ax1.set_xlabel('Datasets')
ax1.set_ylabel('Concept Consistency')
ax1.set_title('Concept Consistency across Datasets')
ax1.set_xticks(index + width)
ax1.set_xticklabels(datasets)
ax1.legend()

# Bar chart for Retrieval Performance
for i, model in enumerate(models):
    retrieval = [retrieval_scores[dataset][i] for dataset in datasets]
    ax2.bar(index + i*width, retrieval, width, label=model)

ax2.set_xlabel('Datasets')
ax2.set_ylabel('Retrieval Performance')
ax2.set_title('Retrieval Performance across Datasets')
ax2.set_xticks(index + width)
ax2.set_xticklabels(datasets)
ax2.legend()

plt.tight_layout()
plt.show()

