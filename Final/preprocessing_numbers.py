import pandas as pd
from sklearn.impute import KNNImputer

# Uses KNNImputer to estimate missing GDP and GINI values.

df = pd.read_csv("processed_country_numbers.csv")

features = ["Freedom Score", "Political Stability Index", "GDP", "GINI"]

gdp_missing = df['GDP'].isnull()
gini_missing = df['GINI'].isnull()

X = df[features]

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

X_imputed[gdp_missing, 2] = df.loc[gdp_missing, 'GDP']
X_imputed[gini_missing, 3] = df.loc[gini_missing, 'GINI']

df['GDP'] = X_imputed[:, 2]
df['GINI'] = X_imputed[:, 3]

df.to_csv("processed_country_numbers.csv", index=False)

