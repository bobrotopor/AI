from classification.debug_files.logitstic_regression import LogisticRegressionGD
from pathlib import Path

import pandas as pd
import numpy as np

CSV_PATH = Path(__file__).parent / 'balanced_df.csv'

balanced_df = pd.read_csv(CSV_PATH)

X = balanced_df.drop(['target'], axis=1)
Y = balanced_df['target']

log_classification = LogisticRegressionGD()
steps, errors = log_classification.fit(
    X, Y,  alpha = 5, epsylon = 0.1, max_steps = 1000, Rtype = "LL",
)

print(errors[0])
print(errors[-1])
print(log_classification.a)