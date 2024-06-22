import numpy as np
import random

def k_medoids_pp(X, n_clusters):
    m, n = X.shape
    medoid_indices = [random.randint(0, m - 1)]
    medoids = X[medoid_indices]

    while len(medoids) < n_clusters:
        distances = np.array([min([np.linalg.norm(x - medoid)**2 for medoid in medoids]) for x in X])
        probabilities = distances / np.sum(distances)
        next_medoid_idx = np.random.choice(m, p=probabilities)
        medoid_indices.append(next_medoid_idx)
        medoids = X[medoid_indices]

    return medoids

def k_medoids(X, n_clusters, max_iter=300):
    medoids = k_medoids_pp(X, n_clusters)
    m, n = X.shape
    labels = np.zeros(m)

    for iteration in range(max_iter):
        # Assign clusters
        distances = np.zeros((m, len(medoids)))
        for i, medoid in enumerate(medoids):
            distances[:, i] = np.linalg.norm(X - medoid, axis=1)

        labels = np.argmin(distances, axis=1)

        # Update medoids
        new_medoids = np.copy(medoids)
        for k in range(len(medoids)):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                medoid_cost = np.inf
                for candidate in cluster_points:
                    cost = np.sum(np.linalg.norm(cluster_points - candidate, axis=1))
                    if cost < medoid_cost:
                        medoid_cost = cost
                        new_medoids[k] = candidate

        # Check for convergence
        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    return medoids, labels