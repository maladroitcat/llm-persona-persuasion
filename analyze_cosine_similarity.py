import math
from pathlib import Path
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, t, f_oneway, shapiro, probplot
from sentence_transformers import SentenceTransformer, util
import torch


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"persona_key", "iteration", "response", "word_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    # basic cleanup
    df = df.dropna(subset=["persona_key", "response"]).copy()
    df["persona_key"] = df["persona_key"].astype(str).str.strip().str.lower()
    df["response"] = df["response"].astype(str).str.strip()
    return df


def embed_texts(texts, model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    # normalize=True so cosine ≈ dot product; improves stability
    emb = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb


def compute_deltas(df: pd.DataFrame, emb: torch.Tensor) -> pd.DataFrame:
    personas = sorted(df["persona_key"].unique())
    # indices for each persona
    persona_idx = {p: df.index[df["persona_key"] == p].tolist() for p in personas}

    # precompute centroid per persona
    centroids = {p: emb[idx].mean(dim=0) for p, idx in persona_idx.items()}

    rows = []
    for p in personas:
        idxs = persona_idx[p]
        emb_p = emb[idxs]
        sum_p = emb_p.sum(dim=0)
        n_p = emb_p.shape[0]

        # centroid of "all other" personas
        other_indices = [i for q in personas if q != p for i in persona_idx[q]]
        emb_other = emb[other_indices]
        centroid_other = emb_other.mean(dim=0)

        for i in idxs:
            vec_i = emb[i]
            # leave-one-out centroid for within similarity
            if n_p > 1:
                loo_centroid = (sum_p - vec_i) / (n_p - 1)
                within = util.cos_sim(vec_i, loo_centroid).item()
            else:
                within = float("nan")  # should not happen with 40/group

            between = util.cos_sim(vec_i, centroid_other).item()
            delta = within - between

            rows.append(
                {
                    "persona_key": p,
                    "index": int(i),
                    "iteration": int(df.loc[i, "iteration"]),
                    "word_count": int(df.loc[i, "word_count"]),
                    "within_sim": within,
                    "between_sim": between,
                    "delta_sim": delta,
                }
            )

    out = pd.DataFrame(rows).dropna(subset=["delta_sim"]).reset_index(drop=True)
    return out, centroids


def ci_mean(x: np.ndarray, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    se = s / math.sqrt(n) if n > 1 else float("nan")
    # t critical
    tcrit = t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else float("nan")
    lo = m - tcrit * se if n > 1 else float("nan")
    hi = m + tcrit * se if n > 1 else float("nan")
    return m, (lo, hi), s


def eta_squared_anova(groups: list[np.ndarray]) -> float:
    # eta^2 = SSB / SST
    all_vals = np.concatenate(groups)
    grand = np.mean(all_vals)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    sst = np.sum((all_vals - grand) ** 2)
    return ssb / sst if sst > 0 else float("nan")


def permutation_p_mean_greater(x: np.ndarray, n_iter=10000, random_state=42):
    rng = np.random.default_rng(random_state)
    obs = float(np.mean(x))
    # sign-flip (null centered at 0)
    flips = rng.choice([-1.0, 1.0], size=(n_iter, x.size))
    sims = flips * x
    perms = sims.mean(axis=1)
    p = float(np.mean(perms >= obs))
    return p


def plots(delta_df: pd.DataFrame):
    # Histogram with KDE
    plt.figure()
    plt.hist(delta_df["delta_sim"], bins=30, density=True, alpha=0.7)
    plt.title("Distribution of Δ (within − between cosine similarity)")
    plt.xlabel("Δ similarity")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("figures/delta_histogram.png", dpi=160)
    plt.close()

    # QQ plot
    plt.figure()
    probplot(delta_df["delta_sim"], dist="norm", plot=plt)
    plt.title("QQ Plot of Δ values")
    plt.tight_layout()
    plt.savefig("figures/delta_qqplot.png", dpi=160)
    plt.close()


def centroid_similarity_matrix(centroids: dict[str, torch.Tensor]) -> pd.DataFrame:
    personas = list(centroids.keys())
    mat = np.zeros((len(personas), len(personas)), dtype=float)
    for i, p1 in enumerate(personas):
        for j, p2 in enumerate(personas):
            mat[i, j] = util.cos_sim(centroids[p1], centroids[p2]).item()
    return pd.DataFrame(mat, index=personas, columns=personas)


def main():
    # Load and embed
    df = load_data("data/silver/responses.csv")
    print(f"Loaded {len(df)} rows across {df['persona_key'].nunique()} personas.")
    emb = embed_texts(df["response"].tolist(), model_name="all-MiniLM-L6-v2")

    # Compute per-message within/between/Δ
    delta_df, centroids = compute_deltas(df, emb)
    delta_path = "data/gold/delta_similarity_per_message.csv"
    delta_df.to_csv(delta_path, index=False)
   
    # Basic summaries
    overall_mean, (lo, hi), overall_sd = ci_mean(delta_df["delta_sim"].values)
    print(f"\nOverall Δ mean = {overall_mean:.4f}  (95% CI [{lo:.4f}, {hi:.4f}]), SD={overall_sd:.4f}")

    by_persona = (
        delta_df.groupby("persona_key")["delta_sim"]
        .agg(["count", "mean", "std"])
        .sort_values("mean", ascending=False)
    )
    persona_ci = []
    for p, grp in delta_df.groupby("persona_key"):
        m, (lo, hi), s = ci_mean(grp["delta_sim"].values)
        persona_ci.append({"persona_key": p, "mean": m, "ci_lo": lo, "ci_hi": hi, "sd": s, "n": len(grp)})
    persona_ci_df = pd.DataFrame(persona_ci).set_index("persona_key").loc[by_persona.index]
    by_persona = by_persona.join(persona_ci_df[["ci_lo", "ci_hi"]])
    by_persona_path = "data/gold/delta_summary_by_persona.csv"
    by_persona.to_csv(by_persona_path)

    # Assumption check (normality of Δ)
    sh_stat, sh_p = shapiro(delta_df["delta_sim"].values)
    print(f"\nShapiro–Wilk normality test on Δ: W={sh_stat:.3f}, p={sh_p:.5f}")

    # One-sample t-test: is mean Δ > 0 ?
    t_stat, p_val = ttest_1samp(delta_df["delta_sim"].values, 0.0, alternative="greater")
    # Cohen's d for one-sample = mean / sd
    d = overall_mean / overall_sd if overall_sd > 0 else float("nan")
    print(f"One-sample t-test (Δ > 0): t={t_stat:.3f}, p={p_val:.5f}, Cohen's d={d:.3f}")

    # Permutation p-value for mean Δ
    p_perm = permutation_p_mean_greater(delta_df["delta_sim"].values, n_iter=1000)
    print(f"Permutation p-value (mean Δ > 0) with 1000 iters: p={p_perm:.5f}")

    # ANOVA: do Δ differ by persona?
    groups = [g["delta_sim"].values for _, g in delta_df.groupby("persona_key")]
    F, pA = f_oneway(*groups)
    eta2 = eta_squared_anova(groups)
    print(f"\nOne-way ANOVA on Δ by persona: F={F:.3f}, p={pA:.5f}, eta^2={eta2:.3f}")

    # Plots (histogram & QQ)
    plots(delta_df)

    # Persona centroid similarity matrix
    sim_df = centroid_similarity_matrix(centroids)
    sim_path = "data/gold/persona_centroid_similarity_matrix.csv"
    sim_df.to_csv(sim_path)

    # Heatmap viz
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        sim_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Cosine similarity"},
        linewidths=0.5,
    )
    plt.title("Persona Centroid-to-Centroid Similarity Matrix")
    plt.xlabel("Persona")
    plt.ylabel("Persona")
    plt.tight_layout()

    heatmap_path = "figures/persona_centroid_similarity_heatmap.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
