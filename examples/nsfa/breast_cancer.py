"""
Example using the breast cancer dataset from Knowles thesis. Data was parsed from the NSFA github repo.
"""
import pandas as pd

import pgfa.models.nsfa

file_name = 'data/breast_cancer.tsv'

df = pd.read_csv(file_name, sep='\t')

print(df.head())

print(df.shape)

print(df.mean(axis=0).shape)

df = (df - df.mean(axis=0))  # / df.std(axis=0)

df = df.T

model = pgfa.models.nsfa.NonparametricSparaseFactorAnalysisModel(df.values)

for i in range(10000):
    if i % 1 == 0:
        print(i, model.params.K, model.rmse, model.log_p)

    model.update(update_type='g', num_particles=20)
