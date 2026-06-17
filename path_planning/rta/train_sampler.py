import pandas as pd
import os

from path_planning.rta.sampling import *

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

def main():
    polar_coords = True
    deterministic = True
    n_estimators = 15
    max_depth = None
    min_samples_leaf = 1
    random_state = 42

    spatial_only = True
    use_her = False
    use_hdg = False
    is_main = True

    data = "deterministic" if deterministic else "stochastic"
    coords = "polar" if polar_coords else "cartesian"

    system = "spatial" if spatial_only else "temporal"
    model_type = "HER" if use_her else "no_HER"

    extra_obs = "_hdg" if use_hdg else ""

    if is_main:
        extra_obs = f"{extra_obs}_main"

    save_path = f"path_planning/rta/data/models/runway_sampler_{data}_{coords}{extra_obs}_{n_estimators}.joblib"

    data_path = f"path_planning/rta/data/{system}/{model_type}{extra_obs}/rta_data_{data}.parquet"
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path, engine="pyarrow") # type: ignore

    required = {"x", "y", "t", "runway", "total_dist_km", "path_len"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check, Filter, and Remove
    col_to_check = 'is_success'
    if col_to_check in df.columns:
        df = df[df[col_to_check]].drop(columns=[col_to_check])

    df["dist_to_go"] = df["total_dist_km"] - df["path_len"]

    df["x_rounded"] = df["x"].round(3)
    df["y_rounded"] = df["y"].round(3)
    df = (
        df.sort_values("dist_to_go", ascending=True) 
        .drop_duplicates(subset=["x_rounded", "y_rounded", "runway"], keep="first")
        .drop(columns=["x_rounded", "y_rounded"])
    )

    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)
    df["theta"] = (np.pi/2 - np.arctan2(df["y"], df["x"])) % (2*np.pi)

    features = ["x", "y"] if not polar_coords else ["r", "theta"]

    print("x:", df["x"].min(), df["x"].max())
    print("y:", df["y"].min(), df["y"].max())

    print("r:", df["r"].min(), df["r"].max())
    print("theta:", df["theta"].min(),df["theta"].max())

    runway_ids = df["runway"].astype("category").values
    X = df[features].values
    y = df["dist_to_go"].values

    print("Creating runway sampler...")
    geo_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    runway_sampler = GeoRunwaySampler(runway_geo=RUNWAYS_SCHIPHOL_FAF, **geo_kwargs)

    print("Fitting runway sampler...")
    runway_sampler.fit(X, y, runway_ids)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving runway sampler to {save_path}")
    runway_sampler.save(save_path)

if __name__ == "__main__":
    main()