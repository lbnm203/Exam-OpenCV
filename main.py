import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load MNIST data
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    data, target = mnist.data, mnist.target
    return data, target

# Hàm hiển thị ảnh
def plot_mnist_images(images, labels, n_images=10):
    plt.figure(figsize=(10, 5))
    for i in range(n_images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    st.pyplot(plt)

# Perform PCA for dimensionality reduction
@st.cache_data
def reduce_dimensions(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

# Plot clusters
def plot_clusters(data, labels, title="Cluster Visualization"):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:  # Noise
            col = [0, 0, 0, 1]  # Màu đen cho noise
            cluster_label = "Noise"
        else:
            cluster_label = f"Cluster {k}"

        class_member_mask = (labels == k)
        xy = data[class_member_mask]

        plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=10, label=cluster_label)

    plt.title(title)
    plt.legend(markerscale=2, fontsize='small', loc='upper right', frameon=True)
    plt.grid(True)
    st.pyplot(plt)

def display_cluster_images(labels, target, data, n_samples=10):
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        st.subheader(f"Cluster {cluster}")
        cluster_indices = np.where(labels == cluster)[0]
        selected_indices = np.random.choice(cluster_indices, min(n_samples, len(cluster_indices)), replace=False)
        for idx in selected_indices:
            image = data[idx].reshape(28, 28)
            st.image(image, caption=f"Label: {target[idx]}")




def run():
    # Main Streamlit app
    st.title("Thuật toán DBSCAN phát hiện các cụm và điểm nhiễu")
    st.sidebar.header("DBSCAN Parameters")
    epsilon = st.sidebar.slider("Epsilon", 0.1, 5.0, 1.0, 0.1)
    min_samples = st.sidebar.slider("Minimum Points (minPts)", 1, 20, 5, 1)

    st.subheader('1. Tập dữ liệu MNIST')
    st.markdown("- Tập dữ liệu MNIST chứa 70.000 hình ảnh số viết tay từ 0 đến 9 với kích thước: 28x28 pixel (grayscale).")

    # Load data
    data, target = load_data()
    reduced_data = reduce_dimensions(data)

    plot_mnist_images(data, target, n_images=10)

    # DBSCAN clustering
    st.subheader("2. DBSCAN Clustering")
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(reduced_data)
    plot_clusters(reduced_data, dbscan_labels, "DBSCAN Clustering")

    # Compute silhouette score for DBSCAN
    if len(set(dbscan_labels)) > 1:
        silhouette_dbscan = silhouette_score(reduced_data, dbscan_labels)
        st.write(f"Silhouette Score for DBSCAN: {silhouette_dbscan:.2f}")
    else:
        st.write("DBSCAN không xác định được cụm. Điều chỉnh epsilon hoặc minPts.")

    # K-Means clustering for comparison
    st.header("K-Means Clustering")
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3, 1)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(reduced_data)
    plot_clusters(reduced_data, kmeans_labels, "K-Means Clustering")

    # Compute silhouette score for K-Means
    silhouette_kmeans = silhouette_score(reduced_data, kmeans_labels)
    st.write(f"Silhouette Score for K-Means: {silhouette_kmeans:.2f}")

    # Compare results
    st.header("Cluster Comparison")
    st.write("""
    - **DBSCAN**: Hoạt động tốt đối với các cụm không tuyến tính, có hình dạng không đều.
    - **K-Means**: Giả sử các cụm có hình cầu và có kích thước bằng nhau.
    """)

    # st.header("Cluster Images (DBSCAN)")
    # display_cluster_images(dbscan_labels, target, data)

    # st.header("Cluster Images (K-Means)")
    # display_cluster_images(kmeans_labels, target, data)

    # comparison_data = pd.DataFrame({
    #     "Method": ["DBSCAN", "K-Means"],
    #     "Clusters": [len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0), k],
    #     "Noise Points": [np.sum(dbscan_labels == -1), 0],
    #     "Silhouette Score": [silhouette_dbscan if 'silhouette_dbscan' in locals() else "N/A", silhouette_kmeans]
    # })
    # st.dataframe(comparison_data)

run()
