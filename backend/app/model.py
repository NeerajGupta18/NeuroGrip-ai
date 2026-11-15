"""
model.py

The grip classifier: a Gaussian classifier with a shared covariance matrix
(linear discriminant analysis), written out in NumPy so every step is visible.

Assumption: for a given person, the 12-number feature vector x of a window
of grasp c is Gaussian with its own mean mu_c but one covariance Sigma shared
by all grasps.  Bayes' rule then gives a score that is *linear* in x:

    score_c(x) = x^T P mu_c  -  0.5 * mu_c^T P mu_c  +  log prior_c ,   P = Sigma^-1

and the posterior over grasps is softmax(score).  A whole recording is
classified by averaging the window scores.

Why this model (see docs/EVALUATION.md for the numbers):
  * It is as accurate as SVM / random forest on this data within a person.
  * Personalising it needs only new class means, so one repetition per grasp
    is enough.  The covariance P (how features co-vary, i.e. the distance
    metric) transfers between people; the class means do not.
  * Because the score is linear, the decision "A rather than B" decomposes
    exactly into one contribution per feature - that is the explanation the
    dashboard shows, with nothing approximated.
"""

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES

GRASPS = ["cyl", "hook", "tip", "palm", "spher", "lat"]
GRASP_LABELS = {
    "cyl": "Cylindrical",
    "hook": "Hook",
    "tip": "Tip pinch",
    "palm": "Palmar",
    "spher": "Spherical",
    "lat": "Lateral",
}
GRASP_DESCRIPTIONS = {
    "cyl": "holding a cylindrical tool, e.g. a bottle",
    "hook": "supporting a heavy load, e.g. a bag handle",
    "tip": "holding a small object between fingertips",
    "palm": "grasping with the palm facing the object",
    "spher": "holding a ball-shaped object",
    "lat": "holding a thin flat object, e.g. a key or card",
}

# Prior strength, in windows, of the population mean when personalising.
# Small on purpose: the evaluation shows other people's class means are a
# poor guide to a new wearer, so the wearer's own data should take over fast.
PRIOR_WINDOWS = 5.0


def class_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([X[y == c].mean(axis=0) for c in range(len(GRASPS))])


def pooled_covariance(X: np.ndarray, y: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Within-class covariance pooled over grasps, shrunk toward a scaled
    identity so it is always safely invertible."""
    residuals = np.concatenate([X[y == c] - X[y == c].mean(axis=0) for c in range(len(GRASPS))])
    sigma = residuals.T @ residuals / len(residuals)
    target = np.eye(sigma.shape[0]) * np.trace(sigma) / sigma.shape[0]
    return (1 - shrinkage) * sigma + shrinkage * target


def personalised_means(population_means: np.ndarray, feature_sums: np.ndarray, counts: np.ndarray,
                       prior_windows: float = PRIOR_WINDOWS) -> np.ndarray:
    """MAP estimate of each grasp's mean for one wearer.

        mu_c = (prior_windows * population_mu_c + sum of the wearer's windows) / (prior_windows + n_c)

    With no data (n_c = 0) this is the population mean; as the wearer confirms
    more recordings it converges to their own average.
    """
    counts = np.asarray(counts, dtype=float)[:, None]
    return (prior_windows * population_means + feature_sums) / (prior_windows + counts)


def window_scores(X: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """(n, F) features -> (n, n_grasps) linear discriminant scores (equal priors)."""
    projected = means @ precision                                # (C, F)
    offsets = 0.5 * np.einsum("cf,cf->c", projected, means)      # 0.5 * mu^T P mu
    return X @ projected.T - offsets


def softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Prediction:
    grasp: str
    runner_up: str
    confidence: float
    confidence_by_grasp: dict
    window_votes: list            # predicted grasp for each window, in time order
    mean_features: np.ndarray     # (F,) features averaged over windows
    contributions: list           # per-feature evidence for grasp vs runner_up
    margin: float                 # score(grasp) - score(runner_up)


def explain(mean_features: np.ndarray, means: np.ndarray, precision: np.ndarray,
            winner: int, rival: int) -> tuple[list, float]:
    """Exact per-feature decomposition of "winner rather than rival".

        score_w - score_r = sum_f  w_f * (x_f - m_f)
        with  w = P (mu_w - mu_r)   and   m = (mu_w + mu_r) / 2

    Each term is one feature's push toward the winner (positive) or toward the
    rival (negative); the terms add up to the margin exactly.
    """
    weights = precision @ (means[winner] - means[rival])
    midpoint = 0.5 * (means[winner] + means[rival])
    terms = weights * (mean_features - midpoint)
    contributions = [
        {
            "feature": name,
            "value": round(float(mean_features[i]), 4),
            "midpoint": round(float(midpoint[i]), 4),
            "contribution": round(float(terms[i]), 4),
        }
        for i, name in enumerate(FEATURE_NAMES)
    ]
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions, float(terms.sum())


def predict(X: np.ndarray, means: np.ndarray, precision: np.ndarray,
            evidence_scale: float = 1.0) -> Prediction:
    """Classify one recording from its (n_windows, F) feature matrix.

    Window scores are averaged, then multiplied by `evidence_scale` before the
    softmax. Summing all windows would treat 23 heavily overlapping windows as
    23 independent observations (wildly overconfident); a plain average treats
    them as one (underconfident). The scale in between is fitted on held-out
    wearers by train.py so that "80% confident" means right about 80% of the time.
    """
    scores = window_scores(X, means, precision)
    recording_scores = scores.mean(axis=0)          # average evidence per window
    probs = softmax(evidence_scale * recording_scores)
    order = np.argsort(recording_scores)[::-1]
    winner, rival = int(order[0]), int(order[1])
    mean_features = X.mean(axis=0)
    contributions, margin = explain(mean_features, means, precision, winner, rival)
    return Prediction(
        grasp=GRASPS[winner],
        runner_up=GRASPS[rival],
        confidence=round(float(probs[winner]), 4),
        confidence_by_grasp={g: round(float(p), 4) for g, p in zip(GRASPS, probs)},
        window_votes=[GRASPS[i] for i in scores.argmax(axis=1)],
        mean_features=mean_features,
        contributions=contributions,
        margin=round(margin, 4),
    )
