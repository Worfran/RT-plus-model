from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.stats import normaltest, lognorm, norm, chi2


#Bright Field data
csv_path = Path(__file__).resolve().parent.parent / "Data" / "RT-BF.csv"
df_bf = pd.read_csv(csv_path)

# Plot all columns in the same figure for Bright Field
numeric_cols_bf = []
col_values = {}
for column in df_bf.columns:
    values = pd.to_numeric(df_bf[column], errors="coerce").dropna()
    if values.empty:
        continue
    numeric_cols_bf.append(column)
    col_values[column] = values

if numeric_cols_bf:
    # determine common bin edges using combined data
    all_values = pd.concat(list(col_values.values()))
    n_bins = max(1, int(math.sqrt(len(all_values))))
    bins = np.histogram_bin_edges(all_values, bins=n_bins)

    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('tab10')
    datasets = [col_values[column] for column in numeric_cols_bf]
    plt.hist(
        datasets,
        bins=bins,
        stacked=True,
        label=numeric_cols_bf,
        color=[colors(i % 10) for i in range(len(numeric_cols_bf))],
        edgecolor='black',
    )

    plt.title("Distribution of loops (Bright Field)")
    plt.xlabel("Size")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    #Plotting histograms with normal distribution fit and lognormal distribution fit
    for column in numeric_cols_bf:
        values = col_values[column]
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
        x = np.linspace(values.min(), values.max(), 200)

        mu, sigma = norm.fit(values)
        _, p_nd = normaltest(values)
        plt.plot(x, norm.pdf(x, mu, sigma), 'k', linewidth=2, label=f'Normal fit (p={p_nd:.4e})')

        positive_values = values[values > 0]
        if not positive_values.empty:
            shape, loc, scale = lognorm.fit(positive_values)
            x_log = np.linspace(positive_values.min(), positive_values.max(), 200)
            #goodness of fit test for lognormal distribution
            Y = values - loc
            Y = Y[Y > 0]
            stat, p_ln = normaltest(np.log(Y))
            print(f"Bright Field - {column}: Lognormal fit p-value = {p_ln:.4e}")
            plt.plot(x_log, lognorm.pdf(x_log, shape, loc=loc, scale=scale), 'r', linewidth=2, label=f'Lognormal fit (p={p_ln:.4e})')

        plt.title(f"Histogram of {column} (Bright Field)")
        plt.xlabel("Size")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

#Dark Field data
csv_path = Path(__file__).resolve().parent.parent / "Data" / "RT-DF.csv"
df_df = pd.read_csv(csv_path)

# Plot all columns in the same figure for Dark Field
numeric_cols_df = []
col_values = {}
for column in df_df.columns:
    values = pd.to_numeric(df_df[column], errors="coerce").dropna()
    if values.empty:
        continue
    numeric_cols_df.append(column)
    col_values[column] = values

if numeric_cols_df:
    # determine common bin edges using combined data
    all_values = pd.concat(list(col_values.values()))
    n_bins = max(1, int(math.sqrt(len(all_values))))
    bins = np.histogram_bin_edges(all_values, bins=n_bins)

    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('tab10')
    datasets = [col_values[column] for column in numeric_cols_df]
    plt.hist(
        datasets,
        bins=bins,
        stacked=True,
        label=numeric_cols_df,
        color=[colors(i % 10) for i in range(len(numeric_cols_df))],
        edgecolor='black',
    )

    plt.title("Distribution of loops (Dark Field)")
    plt.xlabel("Size")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    #Plotting histograms with normal distribution fit
    for column in numeric_cols_df:
        values = col_values[column]
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')
        x = np.linspace(values.min(), values.max(), 200)

        mu, sigma = norm.fit(values)
        _, p_nd = normaltest(values)
        plt.plot(x, norm.pdf(x, mu, sigma), 'k', linewidth=2, label=f'Normal fit (p={p_nd:.4e})')

        positive_values = values[values > 0]
        if not positive_values.empty:
            shape, loc, scale = lognorm.fit(positive_values)
            x_log = np.linspace(positive_values.min(), positive_values.max(), 200)
            #goodness of fit test for lognormal distribution
            Y = values - loc
            Y = Y[Y > 0]
            stat, p_ln = normaltest(np.log(Y))
            print(f"Dark Field - {column}: Lognormal fit p-value = {p_ln:.4e}")
            plt.plot(x_log, lognorm.pdf(x_log, shape, loc=loc, scale=scale), 'r', linewidth=2, label=f'Lognormal fit (p={p_ln:.4e})')

        plt.plot(x, norm.pdf(x, mu, sigma), 'k', linewidth=2)
        plt.title(f"Histogram of {column} (Dark Field)")
        plt.xlabel("Size")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

    plt.show()


