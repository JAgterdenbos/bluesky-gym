import pandas as pd
from sklearn.model_selection import train_test_split

from path_planning.rta.sampling import *

MAX_TIME = 6 * 3600 #6 hours in seconds

def main():
    deterministic = True
    extra_trees = True
    n_estimators = 100
    random_state = 42

    data_path = "path_planning/rta/data/rta_data_undeterministic.parquet"
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path, engine="pyarrow") # type: ignore

    df["rta_remaining"] = df["rta"] - df["t"]
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)
    df["theta"] = (np.pi/2 - np.arctan2(df["y"], df["x"])) % (2*np.pi) # North = 0°

    features = ["x", "y"] if extra_trees else ["r", "theta"]

    unique_runways = df["runway"].unique().astype("category")
    runway_ids = df["runway"].astype("category").values
    X = df[features].values
    y = df["rta_remaining"].values

    # Split all three together to keep indices synced
    X_train, X_test, y_train, y_test, runway_train, runway_test = train_test_split(
        X, y, runway_ids, test_size=0.20, random_state=random_state
    )

    # Fit per-runway interpolators
    interpolators = {}
    for rwy in unique_runways:
        mask = runway_train == rwy
        interp = MinTimeInterpolator(k_neighbors=15)
        interp.fit(X_train[mask], y_train[mask])
        interpolators[rwy] = interp
    
    def min_time_fn(X: np.ndarray, runway_id: Any) -> np.ndarray:
        interp = interpolators.get(runway_id)
        if interp is None:
            raise RuntimeError(f"No interpolator fitted for runway {runway_id}")
        result = interp(X)
        return result

    print("Creating runway sampler...")
    runway_sampler = create_runway_sampler(
        deterministic=deterministic, 
        use_extra_trees=extra_trees,
        n_estimators=n_estimators,
        random_state=random_state,
        min_time_fn=min_time_fn
    )

    print("Fitting runway sampler...")
    runway_sampler.fit(X_train, y_train, runway_train)

    print("Sampling...")
    samples = runway_sampler.sample(X_test, runway_test)
    
    print(np.min(samples))
    error = (samples - y_test)

    error *= MAX_TIME
    
    print(np.abs(error).mean())
    print(np.abs(error).max())
    print(np.abs(error).min())
    
if __name__ == "__main__":
    main()