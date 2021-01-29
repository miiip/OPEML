from pyope.ope import OPE
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import cv2
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.spatial.distance import directed_hausdorff
from random import randint
import glob
import sys

random_key = OPE.generate_key()
cipher = OPE(random_key)
factor = 1<<10

def encrypt(x):
    rnd = random.randint(0, factor - 1)
    x = x*factor + rnd
    x = cipher.encrypt(x)
    return x

def decrypt(x):
    x = cipher.decrypt(x)
    x = x//factor
    return x

def encrypt_image(img):
    img_enc = img.copy().astype(int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                tmp = int(img[i][j][k])
                img_enc[i][j][k] = encrypt(tmp)
    return img_enc

def decrypt_image(img):
    img_dec = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                tmp = int(img[i][j][k])
                img_dec[i][j][k] = decrypt(tmp)
    return img_dec

def generate_random_centroids(number_centroids):
    centers = []
    for i in range(number_centroids):
        x = randint(-(1<<8), 1<<8)
        y = randint(-(1<<8), 1<<8)
        centers.append([x,y])
    return centers

def closestPoint(point, points):
    min = np.linalg.norm(points[0]-point)
    ans = points[0]
    for i in range(1, len(points)):
        d = np.linalg.norm(points[i]-point)
        if d < min:
            min = d
            ans = points[i]
    return [int(coord) for coord in ans]

def reduce_colors(img, num_of_reduced_colors):
    numOfPixels = img.shape[0] * img.shape[1]
    X = np.reshape(img, (numOfPixels, 3))
    centroids, labels = modified_kmeans(X, num_of_reduced_colors)
    k_means_centers = []
    for i in range(num_of_reduced_colors):
        tmp = closestPoint(centroids[i], X)
        k_means_centers.append((tmp[0], tmp[1], tmp[2]))
    reducedX = np.asanyarray([k_means_centers[labels[i]] for i in range(numOfPixels)]).astype(int)
    imgR = np.reshape(reducedX, img.shape)
    return imgR

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def modified_kmeans(X, n_clusters):
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    k_means.fit(X)
    centroids = k_means.cluster_centers_.tolist()
    centroids = [[int(coord) for coord in centroid] for centroid in centroids]
    labels = k_means.labels_.tolist()
    for i in range(n_clusters):
        centroids[i] = closestPoint(centroids[i], X)
    return centroids, labels

def clean_kmeans(X, n_clusters):
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    k_means.fit(X)
    centroids = k_means.cluster_centers_.tolist()
    labels = k_means.labels_.tolist()
    return centroids, labels

def performance_vs_cluster_std():
    results = []
    for cluster_std in range(1, 100, 5):
        X, y = make_blobs(n_samples=100, centers=[[0, 0], [1000, 1000], [-1000, 1000], [1000, -1000]], cluster_std=cluster_std)
        X = X.astype(int)
        y = y.astype(int)
        plain_modified_centroids, plain_labels = modified_kmeans(X, 4)
        X_enc = np.ndarray(shape=X.shape, dtype=np.int)
        for i in range(len(X)):
            valX = int(X[i][0])
            valY = int(X[i][1])
            X_enc[i][0] = encrypt(valX)
            X_enc[i][1] = encrypt(valY)
        enc_centroids, enc_labels = modified_kmeans(X_enc, 4)
        decrypted_centroids = []
        for i in range(len(enc_centroids)):
            decrypted_centroids.append((decrypt(enc_centroids[i][0]), decrypt(enc_centroids[i][1])))
        performance = get_distance_between_two_sets_of_centroids(plain_modified_centroids, decrypted_centroids)
        results.append((cluster_std, performance))
    return results

'''
def get_distance_between_two_sets_of_centroids(centroids1, centroids2):
    num_of_centroids = len(centroids1)
    centroids1 = np.array(centroids1)
    centroids2 = np.array(centroids2)
    viz = [False] * num_of_centroids
    sum = 0
    for centroid in centroids1:
        min = 10 ** 32
        idx = -1
        for i in range(num_of_centroids):
            if viz[i] == False:
                d = np.linalg.norm(centroid - centroids2[i])
                if d < min:
                    min = d
                    idx = i
        if idx == -1:
            print("Somthing went wrong comparing two sets of centroids")
            return 0
        sum = sum + min
        viz[idx] = True
    return sum / num_of_centroids

'''
def encrypt_point(point):
    x = int(point[0])
    y = int(point[1])
    enc_x = encrypt(x)
    enc_y = encrypt(y)
    return [enc_x, enc_y]

def encrypt_set_of_points(points):
    enc_points = []
    for point in points:
        enc_point = encrypt_point(point)
        enc_points.append(enc_point)
    return enc_points

def decrypt_point(point):
    x = point[0]
    y = point[1]
    dec_x = decrypt(x)
    dec_y = decrypt(y)
    return [dec_x, dec_y]

def decrypt_set_of_points(points):
    dec_points = []
    for point in points:
        dec_point = decrypt_point(point)
        dec_points.append(dec_point)
    return dec_points

def get_distance_between_two_sets_of_centroids(centroids1, centroids2):
    centroids1 = np.array(centroids1)
    centroids2 = np.array(centroids2)
    return directed_hausdorff(centroids1, centroids2)[0]

def compare_kmeans_labels_performance(labels_true, labels_pred):
    return(metrics.adjusted_rand_score(labels_true, labels_pred))

#We study the performance of original k-means
#vs our modified version of k-means
def experiment1():
    print("Running experiment 1")
    hausdorff = []
    ari = []
    n_clusters_range = []
    for i in range(1, 101):
        n_clusters = i*10
        n_clusters_range.append(n_clusters)
        centers = generate_random_centroids(n_clusters)
        X, y = make_blobs(n_samples=5000, centers = centers, cluster_std = 1/(0.02*n_clusters))
        k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
        k_means.fit(X)
        original_centroids = k_means.cluster_centers_.tolist()
        original_centroids = [[int(coord) for coord in centroid] for centroid in original_centroids]
        original_labels = k_means.labels_.tolist()

        modified_centroids, modified_labels = modified_kmeans(X, n_clusters)

        hausdorff.append(get_distance_between_two_sets_of_centroids(original_centroids, modified_centroids))
        ari.append(compare_kmeans_labels_performance(original_labels, modified_labels))

    plt.scatter(n_clusters_range, hausdorff, marker='.')
    plt.title('Experiment 1: Hausdorff distance (cluster standard deviation adjusted)')
    plt.xlabel('Number of clusters')
    plt.ylabel('Hausdorff distance')
    plt.show()

    plt.scatter(n_clusters_range, ari, marker='.')
    plt.title('Experiment 1: ARI (cluster standard deviation adjusted)')
    plt.xlabel('Number of clusters')
    plt.ylabel('ARI')
    plt.show()
    return

#encrypted k-means
def experiment2():
    print("Running experiment 2")
    hausdorff = []
    ari = []
    n_clusters_range = []
    for i in range(1, 101):
        n_clusters = i
        n_clusters_range.append(n_clusters)
        centers = generate_random_centroids(n_clusters)
        X, y = make_blobs(n_samples=1000, centers = centers, cluster_std = 1/(0.02*n_clusters))
        X_enc = encrypt_set_of_points(X)
        k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
        k_means.fit(X)

        clean_centroids = k_means.cluster_centers_.tolist()
        clean_centroids = [[int(coord) for coord in centroid] for centroid in clean_centroids]
        clean_labels = k_means.labels_.tolist()

        X_enc = np.array(X_enc)
        encrypted_centroids, encrypted_labels = modified_kmeans(X_enc, n_clusters)
        decrypted_centroids = decrypt_set_of_points(encrypted_centroids)

        hausdorff.append(get_distance_between_two_sets_of_centroids(clean_centroids, decrypted_centroids))
        ari.append(compare_kmeans_labels_performance(clean_labels, encrypted_labels))

    plt.scatter(n_clusters_range, hausdorff, marker='.')
    plt.title('Experiment 2: Hausdorff distance')
    plt.xlabel('Number of clusters')
    plt.ylabel('Hausdorff distance')
    plt.show()

    plt.scatter(n_clusters_range, ari, marker='.')
    plt.title('Experiment 2: ARI')
    plt.xlabel('Number of clusters')
    plt.ylabel('ARI')
    plt.show()
    return


#encrypted DBSCAN
def experiment3():
    print("Running experiment 3")
    ari = []
    n_clusters_range = []
    rng = np.random.RandomState()
    transformation = rng.normal(size=(2, 2))
    num_original_clusters = []
    num_dbscan_clusters = []
    num_enc_dbscan_clusters = []
    print(transformation)

    for i in range(1, 101):
        num_original_clusters.append(i)
        n_clusters = i
        n_clusters_range.append(n_clusters)
        centers = generate_random_centroids(n_clusters)
        X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=1 / (0.02 * n_clusters))
        X = np.dot(X, transformation)
        X_scaled = StandardScaler().fit_transform(X)
        X_enc = encrypt_set_of_points(X)
        X_scaled_enc = StandardScaler().fit_transform(X_enc)

        dbscan = DBSCAN(eps=0.08, min_samples=3)

        clean_labels = dbscan.fit_predict(X_scaled)
        encrypted_labels = dbscan.fit_predict(X_scaled_enc)

        ari.append(compare_kmeans_labels_performance(clean_labels, encrypted_labels))
        num_dbscan_clusters.append(len(list(set(clean_labels))))
        num_enc_dbscan_clusters.append(len(list(set(encrypted_labels))))

    plt.scatter(n_clusters_range, ari, marker='.')
    plt.title('Experiment 3: ARI')
    plt.xlabel('Number of clusters')
    plt.ylabel('ARI')
    plt.show()

    plt.scatter(num_original_clusters, num_enc_dbscan_clusters)
    plt.title('Experiment 3: Number of DBSCAN clusters')
    plt.xlabel('Number of original clusters')
    plt.ylabel('Number of DBSCAN clusters')
    plt.show()
    print(num_original_clusters)
    print(num_enc_dbscan_clusters)
    return



#encrypted color reduction
def experiment4():
    print("Running experiment 4")
    filenames = [img for img in glob.glob("images/*.jpg")]

    image_indices = np.random.randint(low=0, high=len(filenames), size=100)
    filenames = [filenames[index] for index in image_indices]
    psnr_org_dec = []
    for idx in range(len(filenames)-67, len(filenames)-35):
        print('AMR: ', idx)
        file = filenames[idx]
        print(file)
        img = cv2.imread(file)
        img = cv2.resize(img, (128,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_reduce = reduce_colors(img, 64)
        img_enc = encrypt_image(img)
        img_enc_reduced = reduce_colors(img_enc, 64)
        img_dec_reduced = decrypt_image(img_enc_reduced)
        psnr_org_dec.append(psnr(img_dec_reduced, img_reduce))

        img_enc = img_enc % 256
        plt.imshow(img)
        plt.show()

        plt.imshow(img_enc)
        plt.show()

        plt.imshow(img_reduce)
        plt.show()

        plt.imshow(img_dec_reduced)
        plt.show()

        print(psnr(img, img_enc))
        print(psnr(img_dec_reduced, img_reduce))

    print(psnr_org_dec)


print("Enter experiment number (1-4)")
n = len(sys.argv)
if n != 2:
    print("You must eneter a number between 1 and 4")
    quit()

if sys.argv[1] == "1":
    experiment1()
elif sys.argv[1] == "2":
    experiment2()
elif sys.argv[1] == "3":
    experiment3()
elif sys.argv[1] == "4":
    experiment4()
else:
    print("You must eneter a number between 1 and 4")
    quit()

