
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.colors import ListedColormap

RESULTS_DIR = Path('results')
OUT_DIR = Path('analysis_outputs')
OUT_DIR.mkdir(exist_ok=True)
TARGETS = ['encoder_only', 'decoder_only', 'encoder_decoder']


def get_label(raw, target):
    if pd.isna(raw):
        return None
    s = str(raw)
    m = re.search(rf"[\"']{re.escape(target)}[\"']\s*:\s*[\(\[]?\s*([01])\b", s)
    return int(m.group(1)) if m else None


def compute_transferability():
    rows = []
    for fp in sorted(RESULTS_DIR.glob('*.csv')):
        df = pd.read_csv(fp)
        df['success'] = df['success'].astype(str).str.lower().eq('true')
        succ = df[df['success']].copy()

        if len(succ) == 0:
            continue

        attacker = succ['attacker'].dropna().iloc[0]
        source_victim = succ['victim'].dropna().iloc[0]

        row = {
            'file': fp.name,
            'attacker': attacker,
            'source_victim': source_victim,
            'n_success': len(succ)
        }

        for target in TARGETS:
            vals = succ['transfer_preds'].apply(lambda x: get_label(x, target)).dropna()
            row[f'transfer_to_{target}'] = (vals == 0).mean() if len(vals) else np.nan
            row[f'n_eval_{target}'] = len(vals)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(['attacker', 'source_victim'])
    out.to_csv(OUT_DIR / 'table_04_transferability.csv', index=False)
    return out

from matplotlib.colors import LinearSegmentedColormap

def plot_transferability(tab):
    ORDER = ['encoder_only', 'decoder_only', 'encoder_decoder']

    custom_cmap = plt.cm.Blues  # ✅ Blue color scale

    for attacker, g in tab.groupby('attacker'):
        g['source_victim'] = pd.Categorical(
            g['source_victim'],
            categories=ORDER,
            ordered=True
        )
        g = g.sort_values('source_victim')

        cols = [f'transfer_to_{t}' for t in TARGETS]
        mat = g[cols].to_numpy(dtype=float)

        plt.figure(figsize=(6, 4))
        im = plt.imshow(mat, vmin=0, vmax=1, aspect='auto', cmap=custom_cmap)

        plt.colorbar(im, label='Transfer rate (predicted truthful = 0)')
        plt.xticks(range(len(TARGETS)), TARGETS, rotation=20)
        plt.yticks(range(len(g)), g['source_victim'])
        plt.title(f'Transferability matrix ({attacker} attacker)')

        # numbers on cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]

                if np.isnan(val):
                    text = "N/A"
                    color = "black"
                else:
                    text = f"{val:.2f}"
                    color = "white" if val > 0.5 else "black"

                plt.text(j, i, text, ha='center', va='center', color=color)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f'fig_04_transferability_{attacker}.png', dpi=200)
        plt.close()

def main():
    tab = compute_transferability()
    plot_transferability(tab)
    print(tab)


if __name__ == '__main__':
    main()
