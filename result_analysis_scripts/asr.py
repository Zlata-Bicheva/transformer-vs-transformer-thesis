
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import binom_test

RESULTS_DIR = Path('results')
OUT_DIR = Path('analysis_outputs')
OUT_DIR.mkdir(exist_ok=True)


def load_results():
    dfs = []
    for fp in sorted(RESULTS_DIR.glob('*.csv')):
        df = pd.read_csv(fp)
        df['source_file'] = fp.name
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError('No CSV files found in ./results')
    df = pd.concat(dfs, ignore_index=True)
    df['success'] = df['success'].astype(str).str.lower().map({'true': True, 'false': False})
    return df


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, alpha=alpha, method='wilson')
    return float(lo), float(hi)


def asr_summary(df):
    rows = []
    for (attacker, victim), g in df.groupby(['attacker', 'victim']):
        n = len(g)
        k = int(g['success'].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({
            'attacker': attacker,
            'victim': victim,
            'n_attempted': n,
            'n_success': k,
            'ASR': k / n if n else np.nan,
            'ASR_CI_low': lo,
            'ASR_CI_high': hi,
        })
    out = pd.DataFrame(rows).sort_values(['attacker', 'victim'])
    out.to_csv(OUT_DIR / 'table_01_asr_summary.csv', index=False)
    return out

def significance_vs_chance(summary_df, p0=0.5, alpha=0.05):
    rows = []

    for _, r in summary_df.iterrows():
        k = int(r["n_success"])
        n = int(r["n_attempted"])

        if n == 0:
            pval = np.nan
            significant = False
        else:
            # two-sided binomial test
            pval = binom_test(k, n, prop=p0, alternative='two-sided')
            significant = pval < alpha

        rows.append({
            "attacker": r["attacker"],
            "victim": r["victim"],
            "n_attempted": n,
            "n_success": k,
            "ASR": r["ASR"],
            "p_value_vs_0.5": pval,
            "significant_vs_chance": significant
        })

    out = pd.DataFrame(rows).sort_values(["attacker", "victim"])
    out.to_csv(OUT_DIR / "table_03_significance_vs_chance.csv", index=False)

    return out

def plot_asr(summary):
    for attacker, g in summary.groupby('attacker'):
        g = g.sort_values('victim')
        x = np.arange(len(g))
        y = g['ASR'].values
        yerr = np.vstack([y - g['ASR_CI_low'].values, g['ASR_CI_high'].values - y])
        plt.figure(figsize=(7, 4))
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=yerr, fmt='none', capsize=5)
        plt.xticks(x, g['victim'].values, rotation=20)
        plt.ylabel('Attack success rate')
        plt.ylim(0, 1)
        plt.title(f'ASR by victim model ({attacker} attacker)')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'fig_01_asr_{attacker}.png', dpi=200)
        plt.close()

def main():
    df = load_results()
    summary = asr_summary(df)
    sig_table = significance_vs_chance(summary)
    plot_asr(summary)
    print('\nASR summary')
    print(summary)
    print('\nSignificance vs chance (p0 = 0.5)')
    print(sig_table)



if __name__ == '__main__':
    main()
