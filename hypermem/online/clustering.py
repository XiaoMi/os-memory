# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Deterministic spherical topic clustering; timestamps never enter the fit."""
import numpy as np


def unit(vector):
    a = np.asarray(vector, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected a finite, nonempty vector")
    length = float(np.linalg.norm(a))
    return a / length if length > 1e-12 else np.zeros_like(a)


def similarity(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        raise ValueError("Embedding dimensions differ")
    # A zero residual provides no semantic evidence (even against another zero).
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return -1.0
    return float(np.clip(a @ b, -1, 1))


def cohesion(vectors):
    """Mean pairwise cosine in O(n*d), excluding degenerate residuals."""
    x = np.asarray(vectors, dtype=float)
    if x.size == 0:
        return 0.0
    x = x[np.linalg.norm(x, axis=1) > 1e-12]
    n = len(x)
    if n < 2:
        return 1.0 if n else 0.0
    return float(np.clip((np.sum(x,axis=0) @ np.sum(x,axis=0) - np.sum(x*x)) / (n*(n-1)), -1, 1))


def topic_split(vectors, cfg):
    """Fit on a bounded deterministic sample, assign and validate ALL members."""
    x = np.asarray(vectors, dtype=float)
    before = cohesion(x)
    if len(x) < cfg["split_min_members"] or before >= cfg["tau_c"]:
        return None
    if np.any(np.linalg.norm(x,axis=1) < 1e-12):
        raise ValueError("Cluster only nonzero topic-defining representations")
    # Content order makes fitting independent of temporal interleaving.
    order = sorted(range(len(x)), key=lambda i: tuple(x[i]))
    positions = np.linspace(0, len(order)-1, min(len(order),cfg["split_fit_budget"]), dtype=int)
    sample = x[[order[i] for i in positions]]
    distances = np.clip(1.0-sample @ sample.T,0,2)
    np.fill_diagonal(distances,0)
    max_k = min(cfg["split_max_clusters"], len(sample)//cfg["split_min_cluster"])
    best = None
    for k in range(2,max_k+1):
        first = int(np.argmin(sample @ unit(np.mean(sample,axis=0))))
        centers = [sample[first]]
        for _ in range(1,k):
            centers.append(sample[int(np.argmin(np.max(sample @ np.asarray(centers).T,axis=1)))])
        centers = np.asarray(centers)
        previous = None
        for _ in range(cfg["split_iterations"]):
            labels = np.argmax(sample @ centers.T,axis=1)
            if previous is not None and np.array_equal(labels,previous):
                break
            if len(set(labels.tolist())) != k:
                break
            previous = labels.copy()
            centers = np.asarray([unit(np.mean(sample[labels==i],axis=0)) for i in range(k)])
        labels = np.argmax(x @ centers.T,axis=1)
        groups = [np.flatnonzero(labels==i).tolist() for i in range(k)]
        if min(map(len,groups)) < cfg["split_min_cluster"]:
            continue
        within = [cohesion(x[g]) for g in groups]
        after = sum(len(g)*c for g,c in zip(groups,within))/len(x)
        if after-before < cfg["split_min_gain"]-1e-12:
            continue
        fit_labels = np.argmax(sample @ centers.T,axis=1)
        silhouette = []
        for i,label in enumerate(fit_labels):
            own = np.flatnonzero(fit_labels==label)
            if len(own)<2:
                silhouette.append(0.0)
                continue
            a = float(np.sum(distances[i,own])/(len(own)-1))
            alternatives = [float(np.mean(distances[i,fit_labels==other]))
                            for other in range(k) if other!=label and np.any(fit_labels==other)]
            b = min(alternatives) if alternatives else a
            silhouette.append((b-a)/max(a,b) if max(a,b)>1e-12 else 0.0)
        criterion = float(np.mean(silhouette))
        if best is None or criterion > best["criterion"]+1e-12:
            best = {"groups":groups,"cohesion_before":before,"cohesion_after":after,
                    "cluster_cohesions":within,"criterion":criterion,
                    "criterion_kind":"mean_cosine_silhouette_on_fit_sample; singleton silhouette=0",
                    "fit_members":len(sample),"total_members":len(x),
                    "method":"deterministic_spherical_kmeans_not_temporal_segmentation"}
    return best
