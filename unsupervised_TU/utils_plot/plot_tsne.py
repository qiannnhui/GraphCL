from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_embeddings(features, labels, args, epoch, method="PCA"):
    dir = os.path.join("logs", "tsne", args.DS, args.mode, str(args.lr), args.aug)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)  # The exist_ok=True will prevent error if directory exists

    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Method should be either 'PCA' or 't-SNE'")
    
    transformed = reducer.fit_transform(features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=labels, palette="Set1", alpha=0.8)
    plt.title(f"{method} Visualization of ENZYMES Graph Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Class")
    plt.savefig(os.path.join(dir, f"{method}_epoch_{epoch}.png"))