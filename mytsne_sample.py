from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = np.load("db/db_BigBird/db_embeddings.npy")
indices = np.load("db/db_BigBird/train_indices.npy")

metadata=pd.read_csv("./gene-taxa-dataset/metadata.csv")

metadata=metadata.iloc[indices].reset_index(drop=True)

top_classes_gene = metadata["h0-gene"].value_counts().nlargest(10).index.tolist()
metadata=metadata[metadata["h0-gene"].isin(top_classes_gene)]

data=data[list(metadata.index)]
metadata=metadata.reset_index(drop=True)

modelname = "metaberta"

tsne_config = {
    "n_components": 2,
    "n_jobs": 30,
    "perplexity": 40,  
    "max_iter": 1000, 
    "learning_rate": 'auto', 
    "init": 'pca',  
    "random_state": 42,
    "verbose": 1
}


tsne_data = manifold.TSNE(**tsne_config).fit_transform(data)




# Set up a 2x2 grid for the plots
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

nlarge=10
top_classes_gene = metadata["h0-gene"].value_counts().nlargest(nlarge).index.tolist()
top_classes_phylum = metadata["h2-phylum"].value_counts().nlargest(nlarge).index.tolist()


color = "h0-gene"
for g in top_classes_gene:
    s = [i for i, x in enumerate(metadata[color]) if x == g]
    axs[0].scatter(tsne_data[s, 0], tsne_data[s, 1], label=g, marker='o', edgecolors='black', alpha=0.7)

axs[0].set_title("Gene", fontweight='bold')
axs[0].legend()

color = "h2-phylum"
for g in top_classes_phylum:
    s = [i for i, x in enumerate(metadata[color]) if x == g]
    axs[1].scatter(tsne_data[s, 0], tsne_data[s, 1], label=g, marker='o', edgecolors='black', alpha=0.7)

axs[1].set_title("Phylum", fontweight='bold')
axs[1].legend()

# Adjust spacing between subplots
plt.tight_layout()

plt.savefig(modelname +".pdf",dpi=100)
# Show or save the plot as needed
plt.show()