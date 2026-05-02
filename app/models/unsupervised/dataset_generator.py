import numpy as np
import pandas as pd
import os

def generate_dataset(path="app/data/gym_data.csv", n=2000):
    np.random.seed(42)

    # Cluster 1 (bajo)
    x1 = np.random.normal(2, 1, n//3)
    y1 = np.random.normal(300, 70, n//3)

    # Cluster 2 (medio)
    x2 = np.random.normal(7, 1.2, n//3)
    y2 = np.random.normal(650, 90, n//3)

    # Cluster 3 (alto)
    x3 = np.random.normal(12, 1.5, n//3)
    y3 = np.random.normal(900, 120, n//3)

    X = np.concatenate([x1, x2, x3])
    Y = np.concatenate([y1, y2, y3])

    df = pd.DataFrame({
        "hours": X,
        "kcal": Y
    })

    # asegurar carpeta
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Dataset generado en: {path}")
    print(f"Total registros: {len(df)}")


if __name__ == "__main__":
    generate_dataset()