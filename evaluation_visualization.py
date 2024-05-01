import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE

def kmeans_elbow(df, max_k=7):
    sse = {}
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(df)
        sse[k] = kmeans.inertia_

    plt.plot(list(sse.keys()), list(sse.values()), 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for KMeans')
    plt.show()

def kmeans_tsne(df, max_k=7):
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(df)

    kmeans = KMeans(n_clusters=max_k - 1, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X_tsne)
    plt.figure(figsize=(8, 6))
    for i in range(max_k - 1):  
        plt.scatter(X_tsne[y_kmeans == i, 0], X_tsne[y_kmeans == i, 1], label=f'Cluster {i+1}')

    plt.title('K-means Clustering (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()

def kmedoids_elbow(df, max_k=7):
    sse = {}
    for k in range(2, max_k):
        kmedoids = KMedoids(n_clusters=k, random_state=0).fit(df)
        sse[k] = kmedoids.inertia_

    plt.plot(list(sse.keys()), list(sse.values()), 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for KMedoids')
    plt.show()

def kmedoids_tsne(df, max_k=7):
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(df)

    kmedoids = KMedoids(n_clusters=max_k - 1, random_state=0)
    clusters = kmedoids.fit_predict(df)
    plt.figure(figsize=(8, 6))
    for i in range(max_k - 1):  
        plt.scatter(X_tsne[clusters == i, 0], X_tsne[clusters == i, 1], label=f'Cluster {i+1}')

    plt.title('K-medoids Clustering (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()