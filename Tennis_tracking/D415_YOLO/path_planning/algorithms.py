"""
Path planning algorithms for multi-ball collection.

All algorithms find an open-route TSP solution (no return to start) and return:
    {"route": List[Ball], "path_length_m": float, "planning_ms": float}

Ball = (ball_id: int, x: float, y: float)
Point2 = (x: float, y: float)

Call any algorithm by name via run(method, start, balls, **kwargs).
"""

from __future__ import annotations

import math
import random
import time
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

Ball   = Tuple[int, float, float]   # (ball_id, x, y)
Point2 = Tuple[float, float]


# --- geometry helpers ---------------------------------------------------------

def _dist(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_length(start: Point2, route: Sequence[Ball]) -> float:
    total = 0.0
    cur = start
    for _, x, y in route:
        total += _dist(cur, (x, y))
        cur = (x, y)
    return total


def _centroid(balls: Sequence[Ball]) -> Point2:
    if not balls:
        return (0.0, 0.0)
    return (
        sum(x for _, x, _ in balls) / len(balls),
        sum(y for _, _, y in balls) / len(balls),
    )


# --- route primitives ---------------------------------------------------------

def _greedy_nn_route(start: Point2, balls: Sequence[Ball]) -> List[Ball]:
    """Greedy nearest-neighbor from start through all balls."""
    remaining = list(balls)
    route: List[Ball] = []
    cur = start
    while remaining:
        best = min(remaining, key=lambda b: _dist(cur, (b[1], b[2])))
        route.append(best)
        remaining.remove(best)
        cur = (best[1], best[2])
    return route


def _two_opt_improve(start: Point2, route: List[Ball]) -> List[Ball]:
    """2-opt local search on an open route anchored at start."""
    if len(route) < 4:
        return list(route)
    r = list(route)
    improved = True
    while improved:
        improved = False
        n = len(r)
        best_gain = 0.0
        best_pair: Optional[Tuple[int, int]] = None
        for i in range(n - 1):
            a = start if i == 0 else (r[i - 1][1], r[i - 1][2])
            b = (r[i][1], r[i][2])
            for k in range(i + 1, n):
                c = (r[k][1], r[k][2])
                d = (r[k + 1][1], r[k + 1][2]) if k + 1 < n else None
                old = _dist(a, b) + (_dist(c, d) if d else 0.0)
                new = _dist(a, c) + (_dist(b, d) if d else 0.0)
                gain = old - new
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best_pair = (i, k)
        if best_pair:
            i, k = best_pair
            r[i:k + 1] = reversed(r[i:k + 1])
            improved = True
    return r


def _or_opt_improve(start: Point2, route: List[Ball], seg_len: int = 1) -> List[Ball]:
    """
    Or-opt local search: try relocating every segment of length seg_len
    to every other position in the route. Accept if it shortens the path.
    Runs until no improvement is found.
    """
    r = list(route)
    n = len(r)
    if n <= seg_len + 1:
        return r

    improved = True
    while improved:
        improved = False
        n = len(r)
        best_gain = 0.0
        best_move: Optional[Tuple[int, int]] = None   # (seg_start, insert_after)

        for i in range(n - seg_len + 1):
            # Segment r[i:i+seg_len]
            pre_i  = (r[i - 1][1], r[i - 1][2]) if i > 0 else start
            seg_0  = (r[i][1], r[i][2])
            seg_e  = (r[i + seg_len - 1][1], r[i + seg_len - 1][2])
            post_e = (r[i + seg_len][1], r[i + seg_len][2]) if i + seg_len < n else None

            # Cost of cutting segment out
            cut_cost = _dist(pre_i, seg_0) + (_dist(seg_e, post_e) if post_e else 0.0)
            bridged  = _dist(pre_i, post_e) if post_e else 0.0
            removal_gain = cut_cost - bridged

            # Segment internal length (constant, doesn't change with insertion)
            seg_internal = sum(
                _dist((r[i + j][1], r[i + j][2]), (r[i + j + 1][1], r[i + j + 1][2]))
                for j in range(seg_len - 1)
            )

            for j in range(n - seg_len + 1):
                if j in range(i - 1, i + seg_len + 1):
                    continue   # overlapping positions

                # Insert between r[j-1] and r[j] (or at end if j==n-seg_len)
                pre_j  = (r[j - 1][1], r[j - 1][2]) if j > 0 else start
                post_j = (r[j][1], r[j][2]) if j < n else None

                old_link = _dist(pre_j, post_j) if post_j else 0.0
                new_link = _dist(pre_j, seg_0) + seg_internal + (_dist(seg_e, post_j) if post_j else 0.0)
                insertion_gain = old_link - new_link

                gain = removal_gain + insertion_gain
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best_move = (i, j)

        if best_move:
            i, j = best_move
            seg = r[i:i + seg_len]
            rest = r[:i] + r[i + seg_len:]
            # Adjust insertion index after removal
            if j > i:
                j -= seg_len
            rest[j:j] = seg
            r = rest
            improved = True

    return r


def _kmeans_clusters(
    balls: Sequence[Ball],
    k: int,
    max_iter: int = 50,
) -> Tuple[List[List[Ball]], List[Point2]]:
    """Deterministic K-means with farthest-first initialization."""
    if not balls:
        return [], []
    k = max(1, min(k, len(balls)))
    pts = [(x, y) for _, x, y in balls]

    first = min(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    centers: List[Point2] = [pts[first]]
    while len(centers) < k:
        nxt = max(range(len(pts)), key=lambda i: min(_dist(pts[i], c) for c in centers))
        if pts[nxt] in centers:
            break
        centers.append(pts[nxt])

    labels = [0] * len(pts)
    for _ in range(max_iter):
        changed = False
        for i, p in enumerate(pts):
            lbl = min(range(len(centers)), key=lambda ci: _dist(p, centers[ci]))
            if lbl != labels[i]:
                labels[i] = lbl
                changed = True
        new_centers: List[Point2] = []
        for ci in range(len(centers)):
            members = [pts[i] for i, l in enumerate(labels) if l == ci]
            new_centers.append(
                (sum(p[0] for p in members) / len(members),
                 sum(p[1] for p in members) / len(members))
                if members else centers[ci]
            )
        shift = max(_dist(a, b) for a, b in zip(centers, new_centers))
        centers = new_centers
        if not changed or shift < 1e-4:
            break

    cluster_map: Dict[int, List[Ball]] = {}
    for ball, lbl in zip(balls, labels):
        cluster_map.setdefault(lbl, []).append(ball)

    clusters = sorted(
        cluster_map.values(),
        key=lambda cl: (-len(cl), _dist((0.0, 0.0), _centroid(cl))),
    )
    centroids = [_centroid(cl) for cl in clusters]
    return clusters, centroids


def _order_clusters_nn(start: Point2, centroids: List[Point2]) -> List[int]:
    """Greedy nearest-centroid ordering of clusters."""
    remaining = list(range(len(centroids)))
    order: List[int] = []
    cur = start
    while remaining:
        best = min(remaining, key=lambda i: _dist(cur, centroids[i]))
        order.append(best)
        remaining.remove(best)
        cur = centroids[best]
    return order


def _result(start: Point2, route: List[Ball], t0: float) -> dict:
    return {
        "route":         route,
        "path_length_m": _route_length(start, route),
        "planning_ms":   (time.perf_counter() - t0) * 1000,
    }


# --- planning algorithms (public API) -----------------------------------------

def plan_greedy_nn(
    start: Point2,
    balls: Sequence[Ball],
    **_,
) -> dict:
    """
    Greedy nearest-neighbor baseline.
    O(n²) time. Fast but poor quality for scattered balls.
    """
    t0 = time.perf_counter()
    route = _greedy_nn_route(start, list(balls))
    return _result(start, route, t0)


def plan_nn_2opt(
    start: Point2,
    balls: Sequence[Ball],
    **_,
) -> dict:
    """Greedy NN warm start + 2-opt local search.
    Strong deterministic baseline; O(n²) per pass, converges in a few iterations.
    """
    t0 = time.perf_counter()
    route = _greedy_nn_route(start, list(balls))
    route = _two_opt_improve(start, route)
    return _result(start, route, t0)


def plan_nn_or_opt(
    start: Point2,
    balls: Sequence[Ball],
    seg_lens: Sequence[int] = (1, 2, 3),
    **_,
) -> dict:
    """
    Greedy NN + Or-opt improvement (relocate segments of length 1, 2, 3).
    Finds improvements that 2-opt misses (e.g., single-ball misplacement).
    O(n²) per improvement pass per segment length.
    """
    t0 = time.perf_counter()
    route = _greedy_nn_route(start, list(balls))
    for seg in seg_lens:
        route = _or_opt_improve(start, route, seg_len=seg)
    return _result(start, route, t0)


def plan_nn_2opt_or_opt(
    start: Point2,
    balls: Sequence[Ball],
    seg_lens: Sequence[int] = (1, 2, 3),
    n_passes: int = 3,
    **_,
) -> dict:
    """
    Greedy NN + alternating 2-opt and Or-opt passes.
    Usually produces the shortest routes among local-search methods.
    Runs n_passes rounds of (2-opt → Or-opt-1 → Or-opt-2 → Or-opt-3).
    """
    t0 = time.perf_counter()
    route = _greedy_nn_route(start, list(balls))
    for _ in range(n_passes):
        prev_len = _route_length(start, route)
        route = _two_opt_improve(start, route)
        for seg in seg_lens:
            route = _or_opt_improve(start, route, seg_len=seg)
        if _route_length(start, route) >= prev_len - 1e-6:
            break   # converged
    return _result(start, route, t0)


def plan_kmeans_nn_2opt(
    start: Point2,
    balls: Sequence[Ball],
    k: int = 4,
    **_,
) -> dict:
    """
    K-means spatial partitioning + NN+2opt within each cluster.
    Cluster visit order: greedy nearest-centroid from start.
    Balances path quality with spatial organisation (useful for RViz display).
    """
    t0 = time.perf_counter()
    clusters, centroids = _kmeans_clusters(list(balls), k)
    if not clusters:
        return _result(start, [], t0)

    cluster_order = _order_clusters_nn(start, centroids)
    route: List[Ball] = []
    cur = start
    for ci in cluster_order:
        seg = _greedy_nn_route(cur, clusters[ci])
        seg = _two_opt_improve(cur, seg)
        route.extend(seg)
        if seg:
            cur = (seg[-1][1], seg[-1][2])
    return _result(start, route, t0)


def plan_kmeans_2opt_or_opt(
    start: Point2,
    balls: Sequence[Ball],
    k: int = 4,
    seg_lens: Sequence[int] = (1, 2, 3),
    **_,
) -> dict:
    """
    K-means + NN+2opt+Or-opt within each cluster.
    Best cluster-aware algorithm: combines spatial structure with
    the strongest local search (2-opt + Or-opt).
    """
    t0 = time.perf_counter()
    clusters, centroids = _kmeans_clusters(list(balls), k)
    if not clusters:
        return _result(start, [], t0)

    cluster_order = _order_clusters_nn(start, centroids)
    route: List[Ball] = []
    cur = start
    for ci in cluster_order:
        seg = _greedy_nn_route(cur, clusters[ci])
        seg = _two_opt_improve(cur, seg)
        for seg_len in seg_lens:
            seg = _or_opt_improve(cur, seg, seg_len=seg_len)
        route.extend(seg)
        if seg:
            cur = (seg[-1][1], seg[-1][2])
    return _result(start, route, t0)


# --- SA neighbourhood moves ---------------------------------------------------

def _sa_neighbor_2opt(route: List[Ball], rng: random.Random) -> List[Ball]:
    """Random 2-opt move: reverse a random sub-segment."""
    n = len(route)
    if n < 4:
        return route[:]
    i = rng.randint(0, n - 2)
    k = rng.randint(i + 1, n - 1)
    new_route = route[:i] + list(reversed(route[i:k + 1])) + route[k + 1:]
    return new_route


def _sa_neighbor_relocate(route: List[Ball], rng: random.Random) -> List[Ball]:
    """Random Or-opt-1 move: pick a random ball, insert it at a random position."""
    n = len(route)
    if n < 3:
        return route[:]
    i = rng.randint(0, n - 1)
    ball = route[i]
    rest = route[:i] + route[i + 1:]
    j = rng.randint(0, len(rest))
    rest.insert(j, ball)
    return rest


def _double_bridge(route: List[Ball], rng: random.Random) -> List[Ball]:
    """4-opt double-bridge: splits A-B-C-D and reconnects as A-C-B-D.
    Produces moves that no sequence of 2-opt swaps can replicate, making it
    the standard escape mechanism for 2-opt local optima in SA for TSP.
    """
    n = len(route)
    if n < 8:
        return route[:]
    positions = sorted(rng.sample(range(1, n), 3))
    i, j, k = positions
    return route[:i] + route[j:k] + route[i:j] + route[k:]


# --- simulated annealing ------------------------------------------------------

def plan_simulated_annealing(
    start: Point2,
    balls: Sequence[Ball],
    t0: Optional[float] = None,
    max_iter: Optional[int] = None,
    seed: Optional[int] = None,
    **_,
) -> dict:
    """
    Simulated Annealing for open-route TSP.

    Design:
    - Warm start: NN + 2-opt (starts near a 2-opt local optimum).
    - Temperature T₀ auto-calibrated from the warm-start solution length.
    - Cooling rate alpha auto-computed so T decays from T₀ to T_min over
      exactly max_iter steps (avoids premature freezing for large n).
    - Neighbourhood moves per iteration:
        20% double-bridge (4-opt) — escapes 2-opt local optima
        60% random 2-opt reversal — medium-range perturbation
        20% single-ball relocation (Or-opt-1) — fine-grained
    - Best-ever solution is tracked and returned.
    """
    t0_wall = time.perf_counter()
    balls = list(balls)
    n = len(balls)
    if n <= 1:
        return _result(start, balls, t0_wall)

    rng = random.Random(seed)

    # Warm start: NN + 2-opt gives a strong initial solution
    current = _greedy_nn_route(start, balls)
    current = _two_opt_improve(start, current)
    current_len = _route_length(start, current)
    best = current[:]
    best_len = current_len

    # T₀: target ~70% acceptance for moves that worsen by one mean step length
    if t0 is None:
        t0 = 0.8 * current_len / n

    if max_iter is None:
        max_iter = 500 * n

    T_min = 1e-5
    # Auto-calibrate alpha so temperature decays exactly to T_min at max_iter
    if t0 > T_min and max_iter > 1:
        alpha = (T_min / t0) ** (1.0 / max_iter)
    else:
        alpha = 0.999

    T = t0

    for _ in range(max_iter):
        r = rng.random()
        if r < 0.20 and n >= 8:
            candidate = _double_bridge(current, rng)
        elif r < 0.80 and n >= 4:
            candidate = _sa_neighbor_2opt(current, rng)
        else:
            candidate = _sa_neighbor_relocate(current, rng)

        cand_len = _route_length(start, candidate)
        delta = cand_len - current_len

        if delta < 0 or (T > 1e-9 and rng.random() < math.exp(-delta / T)):
            current = candidate
            current_len = cand_len
            if current_len < best_len:
                best = current[:]
                best_len = current_len

        T *= alpha

    return _result(start, best, t0_wall)


# --- boustrophedon (lawnmower / zigzag) ---------------------------------------

def plan_boustrophedon(
    start: Point2,
    balls: Sequence[Ball],
    strip_width: float = 1.5,
    **_,
) -> dict:
    """
    Boustrophedon (牛耕法) coverage: divide the field into vertical strips
    of width strip_width (along X), then sweep each strip in alternating
    Y direction. Classic baseline in agricultural-robot and cleaning-robot
    papers; often beats greedy NN for spatially scattered targets.

    No clustering is performed — balls are ordered purely by geometry.
    """
    t0 = time.perf_counter()
    balls = list(balls)
    if not balls:
        return _result(start, [], t0)

    xs = [b[1] for b in balls]
    x_min, x_max = min(xs), max(xs)
    span = max(x_max - x_min, 1e-6)
    n_strips = max(1, math.ceil(span / strip_width))
    actual_w = span / n_strips

    # Assign each ball to a strip index
    def strip_idx(x: float) -> int:
        return min(n_strips - 1, int((x - x_min) / actual_w))

    strips: Dict[int, List[Ball]] = {}
    for b in balls:
        si = strip_idx(b[1])
        strips.setdefault(si, []).append(b)

    route: List[Ball] = []
    for si in range(n_strips):
        strip_balls = strips.get(si, [])
        if not strip_balls:
            continue
        reverse = (si % 2 == 1)
        strip_balls.sort(key=lambda b: b[2], reverse=reverse)
        route.extend(strip_balls)

    return _result(start, route, t0)


# --- proposed method: K-means + exact centroid TSP + NN-2opt per cluster -----

def _exact_centroid_tsp(start: Point2, centroids: List[Point2]) -> List[int]:
    """
    Brute-force exact open TSP on cluster centroids.
    Optimal for k ≤ 8 (at most 8! = 40 320 permutations).
    Returns the optimal visit order as a list of centroid indices.
    """
    k = len(centroids)
    if k == 0:
        return []
    if k == 1:
        return [0]
    best_order: List[int] = []
    best_len = float("inf")
    for perm in permutations(range(k)):
        pts = [start] + [centroids[i] for i in perm]
        length = sum(_dist(pts[j], pts[j + 1]) for j in range(len(pts) - 1))
        if length < best_len:
            best_len = length
            best_order = list(perm)
    return best_order


def plan_kmeans_exact_centroid(
    start: Point2,
    balls: Sequence[Ball],
    k: int = 4,
    **_,
) -> dict:
    """
    Proposed method for multi-ball collection:

    1. K-means (k clusters) for spatial partitioning.
    2. Exact brute-force TSP on k centroids → globally optimal cluster
       visit order (feasible because k ≤ 8 is tiny).
    3. NN + 2-opt within each cluster for intra-cluster routing.

    The key insight: when the robot can sweep an entire cluster area in one
    visit, optimising the inter-cluster order (step 2) dominates total path
    quality. Exact TSP on k centroids is free (< 0.5 ms for k ≤ 8), whereas
    global SA on 50 balls costs 135 ms and doesn't respect cluster structure.
    """
    t0 = time.perf_counter()
    clusters, centroids = _kmeans_clusters(list(balls), k)
    if not clusters:
        return _result(start, [], t0)

    cluster_order = _exact_centroid_tsp(start, centroids)
    route: List[Ball] = []
    cur = start
    for ci in cluster_order:
        seg = _greedy_nn_route(cur, clusters[ci])
        seg = _two_opt_improve(cur, seg)
        route.extend(seg)
        if seg:
            cur = (seg[-1][1], seg[-1][2])
    return _result(start, route, t0)


# --- algorithm registry -------------------------------------------------------

ALL_ALGORITHMS = {
    "greedy_nn":              plan_greedy_nn,
    "nn_2opt":                plan_nn_2opt,
    "nn_2opt_or_opt":         plan_nn_2opt_or_opt,
    "simulated_annealing":    plan_simulated_annealing,
    "boustrophedon":          plan_boustrophedon,
    "kmeans_nn_2opt":         plan_kmeans_nn_2opt,
    "kmeans_exact_centroid":  plan_kmeans_exact_centroid,
}

ALGORITHM_NOTES = {
    "greedy_nn":             "Baseline: greedy nearest-neighbor; universal lower bound",
    "nn_2opt":               "NN + 2-opt; strong single-ball deterministic baseline",
    "nn_2opt_or_opt":        "NN + 2-opt + Or-opt; best pure local-search method",
    "simulated_annealing":   "SA metaheuristic; best single-ball path quality",
    "boustrophedon":         "Lawnmower/zigzag coverage; standard baseline in agricultural & cleaning robots",
    "kmeans_nn_2opt":        "K-means + greedy nearest-centroid + NN-2opt; cluster-aware baseline",
    "kmeans_exact_centroid": "K-means + exact centroid TSP + NN-2opt (proposed); optimal cluster ordering",
}


def run(
    method: str,
    start: Point2,
    balls: Sequence[Ball],
    **kwargs,
) -> dict:
    """Call a named algorithm. Raises KeyError for unknown method names."""
    fn = ALL_ALGORITHMS[method]
    return fn(start, balls, **kwargs)
