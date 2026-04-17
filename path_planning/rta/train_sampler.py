import pandas as pd
import os

from path_planning.rta.sampling import *

def main():
    deterministic = True
    extra_trees = True
    n_estimators = 100
    random_state = 42

    model_type = "deterministic" if deterministic else "probabilistic"
    tree_type = "extra_trees" if extra_trees else "random_forest"

    save_path = f"path_planning/rta/data/models/runway_sampler_{model_type}_{tree_type}_{n_estimators}.pkl"

    data_path = "path_planning/rta/data/rta_data_undeterministic.parquet"
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path, engine="pyarrow") # type: ignore

    df["rta_remaining"] = df["rta"] - df["t"]
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)
    df["theta"] = (np.pi/2 - np.arctan2(df["y"], df["x"])) % (2*np.pi)

    features = ["x", "y"] if extra_trees else ["r", "theta"]

    print("x:", df["x"].min(), df["x"].max())
    print("y:", df["y"].min(), df["y"].max())

    print("r:", df["r"].min(), df["r"].max())
    print("theta:", df["theta"].min(),df["theta"].max())

    runway_ids = df["runway"].astype("category").values
    X = df[features].values
    y = df["rta_remaining"].values

    print("Creating runway sampler...")
    runway_sampler = create_runway_sampler(
        deterministic=deterministic, 
        use_extra_trees=extra_trees,
        n_estimators=n_estimators,
        random_state=random_state
    )

    print("Fitting runway sampler...")
    runway_sampler.fit(X, y, runway_ids)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving runway sampler to {save_path}")
    runway_sampler.save(save_path)

if __name__ == "__main__":
    main()