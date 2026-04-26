import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from path_planning.rta.sampling import RunwaySpecificSampler, UnifiedRunwaySampler, GeoRunwaySampler

RUNWAYS_SCHIPHOL_FAF = {
    "18C": {"lat": 52.301851, "lon": 4.737557, "track": 183},
    "36C": {"lat": 52.330937, "lon": 4.740026, "track":   3},
    "18L": {"lat": 52.291274, "lon": 4.777391, "track": 183},
    "36R": {"lat": 52.321199, "lon": 4.780119, "track":   3},
    "18R": {"lat": 52.329170, "lon": 4.708888, "track": 183},
    "36L": {"lat": 52.362334, "lon": 4.711910, "track":   3},
    "06":  {"lat": 52.304278, "lon": 4.776817, "track":  60},
    "24":  {"lat": 52.288020, "lon": 4.734463, "track": 240},
    "09":  {"lat": 52.318362, "lon": 4.796749, "track":  87},
    "27":  {"lat": 52.315940, "lon": 4.712981, "track": 267},
    "04":  {"lat": 52.313783, "lon": 4.802666, "track":  45},
    "22":  {"lat": 52.300518, "lon": 4.783853, "track": 225},
}


def model_size(model, name: str):
    import io
    import joblib
    buf = io.BytesIO()
    joblib.dump(model, buf)
    size_bytes = buf.tell()
    print(f"{name}: {size_bytes / 1024:.1f} KB  ({size_bytes / 1024**2:.2f} MB / {size_bytes / 1024**3:.2f} GB)")


def evaluate(name: str, samples: np.ndarray, y_test: np.ndarray):
    error = np.abs(samples - y_test)
    print(f"\n--- {name} ---")
    print(f"  MAE:  {error.mean():.4f}")
    print(f"  RMSE: {np.sqrt((error ** 2).mean()):.4f}")


def main():
    polar_coords = True
    deterministic = True
    n_estimators = 15
    max_depth = None
    min_samples_leaf = 1
    random_state = 42

    data = "deterministic" if deterministic else "stochastic"

    data_path = f"path_planning/rta/data/rta_data_{data}.parquet"
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path, engine="pyarrow")  # type: ignore

    df["dist_to_go"] = df["total_dist_km"] - df["path_len"]

    df["x_rounded"] = df["x"].round(3)
    df["y_rounded"] = df["y"].round(3)
    df = (
        df.sort_values("dist_to_go", ascending=True) 
        .drop_duplicates(subset=["x_rounded", "y_rounded", "runway"], keep="first")
        .drop(columns=["x_rounded", "y_rounded"])
    )

    df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["theta"] = (np.pi / 2 - np.arctan2(df["y"], df["x"])) % (2 * np.pi)

    features = ["x", "y"] if not polar_coords else ["r", "theta"]

    # Use plain string arrays instead of pandas Categorical to avoid
    # mask comparison issues in RunwaySpecificSampler
    runway_ids = df["runway"].astype(str).values
    X = df[features].astype(np.float32).values
    y = df["dist_to_go"].astype(np.float32).values

    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(indices, test_size=0.20, random_state=random_state)

    X_train, X_test           = X[idx_train], X[idx_test]
    y_train, y_test           = y[idx_train], y[idx_test]
    runway_train, runway_test = runway_ids[idx_train], runway_ids[idx_test]

    # ── Approach 1: per-runway models ────────────────────────────────────────
    per_runway_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    print("\nFitting RunwaySpecificSampler...")
    per_runway = RunwaySpecificSampler(**per_runway_kwargs)
    per_runway.fit(X_train, y_train, runway_train)
    samples_per = per_runway.sample(X_test, runway_test)
    evaluate("RunwaySpecificSampler", samples_per, y_test)
    model_size(per_runway, "RunwaySpecificSampler")

    del per_runway  # Free memory before fitting next model

    # ── Approach 2: single unified model, runway as encoded feature ──────────
    unified_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    print("\nFitting UnifiedRunwaySampler...")
    unified = UnifiedRunwaySampler(**unified_kwargs, known_runways=list(RUNWAYS_SCHIPHOL_FAF.keys()))
    unified.fit(X_train, y_train, runway_train)
    samples_uni = unified.sample(X_test, runway_test)
    evaluate("UnifiedRunwaySampler", samples_uni, y_test)
    model_size(unified, "UnifiedRunwaySampler")

    del unified  # Free memory before fitting next model

    # ── Approach 3: single model, runway as geo features (lat/lon/track) ─────
    # Features per sample: [*spatial, faf_lat, faf_lon, sin(track), cos(track)]
    # Track is sin/cos encoded to handle the 359->0 wrap-around.

    geo_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    print("\nFitting GeoRunwaySampler...")
    geo = GeoRunwaySampler(runway_geo=RUNWAYS_SCHIPHOL_FAF, **geo_kwargs)
    geo.fit(X_train, y_train, runway_train)
    samples_geo = geo.sample(X_test, runway_test)
    evaluate("GeoRunwaySampler", samples_geo, y_test)
    model_size(geo, "GeoRunwaySampler")

if __name__ == "__main__":
    main()