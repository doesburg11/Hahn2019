"""Periodic geometry and density-cluster diagnostics (no learning reward)."""
import numpy as np
from .environment import displacement, wrap_angle


def dbscan(distance, eps=4., min_samples=3):
    """DBSCAN from a precomputed distance matrix; self counts toward min_samples."""
    n = len(distance)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    neighborhoods = [np.flatnonzero(row <= eps).tolist() for row in distance]
    cluster = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighborhoods[i]) < min_samples:
            continue
        labels[i] = cluster
        queue = list(neighborhoods[i])
        queued = set(queue)
        for j in queue:
            if not visited[j]:
                visited[j] = True
                if len(neighborhoods[j]) >= min_samples:
                    for k in neighborhoods[j]:
                        if k not in queued:
                            queued.add(k)
                            queue.append(k)
            if labels[j] == -1:
                labels[j] = cluster
        cluster += 1
    return labels


def angle_deviation(headings):
    vector = np.exp(1j*np.asarray(headings)).mean()
    if abs(vector) < 1e-12:
        return None  # Circular mean is undefined for a balanced opposing group.
    return float(np.rad2deg(np.abs(wrap_angle(headings-np.angle(vector)))).mean())


def snapshot_metrics(env, eps=4., min_samples=3):
    c = env.config
    delta = displacement(env.positions[:, None, :], env.positions[None, :, :], c.side)
    distance = np.linalg.norm(delta, axis=-1)
    pairs = np.triu_indices(c.n_prey, 1)
    labels = dbscan(distance, eps, min_samples)
    clusters = [np.flatnonzero(labels == label) for label in sorted(set(labels)-{-1})]
    deviations, weighted_agents, total_distance, total_pairs = 0., 0, 0., 0
    for ids in clusters:
        dev = angle_deviation(env.headings[ids])
        if dev is not None:
            deviations += len(ids)*dev
            weighted_agents += len(ids)
        values = distance[np.ix_(ids, ids)][np.triu_indices(len(ids), 1)]
        total_distance += values.sum()
        total_pairs += len(values)
    noise = np.flatnonzero(labels == -1)
    noise_pairs = distance[np.ix_(noise, noise)][np.triu_indices(len(noise), 1)]
    return {
        "noise_angle_deviation_deg": angle_deviation(env.headings[noise]) if len(noise) else None,
        "noise_pair_distance_over_side": float(noise_pairs.mean()/c.side) if len(noise_pairs) else None,
        "polarization": float(abs(np.exp(1j*env.headings).mean())),
        "mean_angle_deviation_deg": angle_deviation(env.headings),
        "mean_pair_distance_over_side": float(distance[pairs].mean()/c.side),
        "clusters": len(clusters), "noise_fraction": float((labels == -1).mean()),
        "cluster_angle_deviation_deg": deviations/weighted_agents if weighted_agents else None,
        "cluster_pair_distance_over_side": float(total_distance/total_pairs/c.side) if total_pairs else None,
    }
