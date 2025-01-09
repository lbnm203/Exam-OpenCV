# import streamlit as st
# from sklearn.datasets import fetch_openml
# from sklearn.decomposition import PCA
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Load MNIST data
# @st.cache_data
# def load_data():
#     mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#     data, target = mnist.data, mnist.target
#     return data, target

# # Hàm hiển thị ảnh
# def plot_mnist_images(images, labels, n_images=10):
#     plt.figure(figsize=(10, 5))
#     for i in range(n_images):
#         plt.subplot(1, n_images, i + 1)
#         plt.imshow(images[i].reshape(28, 28), cmap='gray')
#         plt.title(f"Label: {labels[i]}")
#         plt.axis('off')
#     st.pyplot(plt)

# # Perform PCA for dimensionality reduction
# @st.cache_data
# def reduce_dimensions(data, n_components=2):
#     pca = PCA(n_components=n_components)
#     reduced_data = pca.fit_transform(data)
#     return reduced_data

# # Plot clusters
# def plot_clusters(data, labels, title="Cluster Visualization"):
#     plt.figure(figsize=(8, 6))
#     unique_labels = np.unique(labels)
#     colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

#     for k, col in zip(unique_labels, colors):
#         if k == -1:  # Noise
#             col = [0, 0, 0, 1]  # Màu đen cho noise
#             cluster_label = "Noise"
#         else:
#             cluster_label = f"Cluster {k}"

#         class_member_mask = (labels == k)
#         xy = data[class_member_mask]

#         plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=10, label=cluster_label)

#     plt.title(title)
#     plt.legend(markerscale=2, fontsize='small', loc='upper right', frameon=True)
#     plt.grid(True)
#     st.pyplot(plt)

# def display_cluster_images(labels, target, data, n_samples=10):
#     unique_clusters = np.unique(labels)
#     for cluster in unique_clusters:
#         st.subheader(f"Cluster {cluster}")
#         cluster_indices = np.where(labels == cluster)[0]
#         selected_indices = np.random.choice(cluster_indices, min(n_samples, len(cluster_indices)), replace=False)
#         for idx in selected_indices:
#             image = data[idx].reshape(28, 28)
#             st.image(image, caption=f"Label: {target[idx]}")


# def run():
#     # Main Streamlit app
#     st.title("Thuật toán DBSCAN phát hiện các cụm và điểm nhiễu")
#     st.sidebar.header("DBSCAN Parameters")
#     epsilon = st.sidebar.slider("Epsilon", 0.1, 5.0, 1.0, 0.1)
#     min_samples = st.sidebar.slider("Minimum Points (minPts)", 1, 20, 5, 1)

#     st.subheader('1. Tập dữ liệu MNIST')
#     st.markdown("- Tập dữ liệu MNIST chứa 70.000 hình ảnh số viết tay từ 0 đến 9 với kích thước: 28x28 pixel (grayscale).")

#     # Load data
#     data, target = load_data()
#     reduced_data = reduce_dimensions(data)

#     plot_mnist_images(data, target, n_images=10)

#     # DBSCAN clustering
#     st.subheader("2. DBSCAN Clustering")
#     dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
#     dbscan_labels = dbscan.fit_predict(reduced_data)
#     plot_clusters(reduced_data, dbscan_labels, "DBSCAN Clustering")

#     # Compute silhouette score for DBSCAN
#     if len(set(dbscan_labels)) > 1:
#         silhouette_dbscan = silhouette_score(reduced_data, dbscan_labels)
#         st.write(f"Silhouette Score for DBSCAN: {silhouette_dbscan:.2f}")
#     else:
#         st.write("DBSCAN không xác định được cụm. Điều chỉnh epsilon hoặc minPts.")

#     # K-Means clustering for comparison
#     st.header("K-Means Clustering")
#     k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3, 1)
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans_labels = kmeans.fit_predict(reduced_data)
#     plot_clusters(reduced_data, kmeans_labels, "K-Means Clustering")

#     # Compute silhouette score for K-Means
#     silhouette_kmeans = silhouette_score(reduced_data, kmeans_labels)
#     st.write(f"Silhouette Score for K-Means: {silhouette_kmeans:.2f}")

#     # Compare results
#     st.header("Cluster Comparison")
#     st.write("""
#     - **DBSCAN**: Hoạt động tốt đối với các cụm không tuyến tính, có hình dạng không đều.
#     - **K-Means**: Giả sử các cụm có hình cầu và có kích thước bằng nhau.
#     """)

#     # st.header("Cluster Images (DBSCAN)")
#     # display_cluster_images(dbscan_labels, target, data)

#     # st.header("Cluster Images (K-Means)")
#     # display_cluster_images(kmeans_labels, target, data)

#     # comparison_data = pd.DataFrame({
#     #     "Method": ["DBSCAN", "K-Means"],
#     #     "Clusters": [len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0), k],
#     #     "Noise Points": [np.sum(dbscan_labels == -1), 0],
#     #     "Silhouette Score": [silhouette_dbscan if 'silhouette_dbscan' in locals() else "N/A", silhouette_kmeans]
#     # })
#     # st.dataframe(comparison_data)

# run()

import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Noise: Black color
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col],
                    label=f"Cluster {k}" if k != -1 else "Noise")
    plt.title(title)
    plt.legend()
    st.pyplot(plt)


def step_by_step_dbscan(data, epsilon, min_samples):
    st.subheader("Step-by-Step DBSCAN")
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

    labels = np.full(data.shape[0], -1)  # Initialize all points as noise
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    unique_labels = set()

    # Process points step by step
    for i, point in enumerate(data):
        neighbors = []
        for j, other_point in enumerate(data):
            if np.linalg.norm(point - other_point) <= epsilon:
                neighbors.append(j)

        # Mark core points if they satisfy min_samples
        if len(neighbors) >= min_samples:
            core_samples_mask[i] = True
            cluster_id = len(
                unique_labels) if i not in unique_labels else labels[i]
            labels[i] = cluster_id
            unique_labels.add(cluster_id)
            for neighbor in neighbors:
                labels[neighbor] = cluster_id

    # Plot results for each cluster in a grid
    st.write("Visualizing clusters in a grid")
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_cols = min(3, num_clusters)
    num_rows = (num_clusters // num_cols) + \
        (1 if num_clusters % num_cols != 0 else 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten() if num_clusters > 1 else [axes]

    for idx, cluster_id in enumerate(unique_labels):
        if cluster_id == -1:
            continue  # Skip noise

        cluster_points = data[labels == cluster_id]
        axes[idx].scatter(cluster_points[:, 0], cluster_points[:,
                          1], s=10, label=f"Cluster {cluster_id}")
        axes[idx].set_title(f"Cluster {cluster_id}")
        axes[idx].legend()

    plt.tight_layout()
    st.pyplot(fig)


def display_cluster_images(labels, target, n_samples=10):
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        st.subheader(f"Cluster {cluster}")
        cluster_indices = np.where(labels == cluster)[0]
        selected_indices = np.random.choice(cluster_indices, min(
            n_samples, len(cluster_indices)), replace=False)
        for idx in selected_indices:
            image = data[idx].reshape(28, 28)
            st.image(image, caption=f"Label: {target[idx]}")


def run():
    # Main Streamlit app
    st.title("Thuật toán DBSCAN phát hiện các cụm và điểm nhiễu")
    st.write("DBSCAN là một thuật toán phân cụm dựa trên mật độ (Density-Based Clustering) được sử dụng phổ biến trong học máy và khai phá dữ liệu. Thuật toán không yêu cầu số lượng cụm phải được định nghĩa trước (như K-Means) và có khả năng xác định các điểm nhiễu (noise).")
    st.write("### Nguyên lý hoạt động")
    st.write("DBSCAN phân cụm dựa trên mật độ của dữ liệu, sử dụng hai tham số chính:")
    st.markdown(
        "- **Epsilon (ε):** Khoảng cách tối đa để xác định các điểm lân cận.")
    st.markdown(
        "- **MinPts:** Số lượng điểm tối thiểu trong vùng lân cận để một điểm được coi là \"điểm lõi\".")
    st.write("Mỗi điểm được phân loại thành ba nhóm:")
    st.markdown("1. **Điểm lõi (Core Point):** Đủ MinPts trong bán kính ε.")
    st.markdown(
        "2. **Điểm biên (Border Point):** Nằm trong vùng lân cận của điểm lõi, nhưng không đủ MinPts.")
    st.markdown(
        "3. **Điểm nhiễu (Noise Point):** Nằm ngoài vùng lân cận của bất kỳ điểm lõi nào.")
    st.sidebar.header("DBSCAN Parameters")
    epsilon = st.sidebar.slider("Epsilon", 0.1, 5.0, 1.0, 0.1)
    min_samples = st.sidebar.slider("Minimum Points (minPts)", 1, 20, 5, 1)

    st.subheader('1. Tập dữ liệu MNIST')
    st.markdown("- Tập dữ liệu MNIST (Modified National Institute of Standards and Technology) là một trong những tập dữ liệu nổi tiếng nhất trong lĩnh vực học máy và thị giác máy tính, đặc biệt được sử dụng rộng rãi cho các bài toán phân loại hình ảnh. Dưới đây là các thông tin chi tiết:")
    st.markdown("- Tập MNIST chứa 70.000 hình ảnh số viết tay từ 0 đến 9:")
    st.markdown("   - Trong đó: 60.000 hình ảnh dùng để huấn luyện, 10.000 hình ảnh dùng để kiểm tra (testing) với kích thước: 28x28 pixel (grayscale). Mỗi hình ảnh được gán một nhãn tương ứng (label), biểu thị số viết tay")

    data, target = load_data()
    plot_mnist_images(data, target, n_images=10)

    # Load data
    reduced_data = reduce_dimensions(data)

    # Step-by-step DBSCAN clustering
    step_by_step_dbscan(reduced_data, epsilon, min_samples)

    # DBSCAN clustering
    st.header("DBSCAN Clustering")
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(reduced_data)
    plot_clusters(reduced_data, dbscan_labels, "DBSCAN Clustering")

    # Compute silhouette score for DBSCAN
    if len(set(dbscan_labels)) > 1:
        silhouette_dbscan = silhouette_score(reduced_data, dbscan_labels)
        st.write(f"Silhouette Score for DBSCAN: {silhouette_dbscan:.2f}")
    else:
        st.write("DBSCAN failed to identify clusters. Adjust epsilon or minPts.")

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
    - **DBSCAN**: Works well for non-linear, irregularly shaped clusters.
    - **K-Means**: Assumes clusters are spherical and equally sized.
    """)

    st.header("Cluster Images (DBSCAN)")
    display_cluster_images(dbscan_labels, target)

    st.header("Cluster Images (K-Means)")
    display_cluster_images(kmeans_labels, target)

    comparison_data = pd.DataFrame({
        "Method": ["DBSCAN", "K-Means"],
        "Clusters": [len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0), k],
        "Noise Points": [np.sum(dbscan_labels == -1), 0],
        "Silhouette Score": [silhouette_dbscan if 'silhouette_dbscan' in locals() else "N/A", silhouette_kmeans]
    })
    st.dataframe(comparison_data)


run()
