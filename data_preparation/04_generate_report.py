# data_preparation/04_generate_report.py
"""
Phase 1A — Step 4: Generate the dataset curation report and paper-ready figures.

Outputs:
  data/processed/report/dataset_report.txt   — text tables for the paper
  data/processed/report/dataset_stats.csv    — machine-readable per-source stats
  data/processed/report/class_distribution.png — Figure for Section 3 (Methodology)

FIGURE DESIGN DECISIONS (documented for reproducibility):
  - The deduplication step produces merged source tags (e.g. "deepset|jasperls")
    for rows whose text appeared in multiple datasets. These are correct data
    but produce unreadable x-axis labels in matplotlib. We canonicalise them
    to a single primary source name for the figure, and document the overlap
    in a footnote and in the paper text. This keeps the figure consistent
    with Table 1 in the report.
  - jasperls is shown as a footnote under deepset rather than a zero-height
    bar, since all jasperls rows were merged into deepset by deduplication.
  - Percentage labels are added inside the split bars so the 58.5/41.5
    ratio is immediately visible without reading the y-axis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH   = "data/processed/merged_raw.csv"
DEDUP_PATH = "data/processed/merged_deduped.csv"
TRAIN_PATH = "data/processed/final_split/train.csv"
VAL_PATH   = "data/processed/final_split/val.csv"
TEST_PATH  = "data/processed/final_split/test.csv"
REPORT_DIR = "data/processed/report"
os.makedirs(REPORT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Source metadata — ground truth numbers from the curation pipeline.
# These are the canonical per-source sample counts used in both the report
# table and the figure. We define them explicitly here (rather than reading
# from the deduped CSV source column) because the dedup step produces merged
# source tags like "deepset|jasperls" that need to be resolved cleanly.
#
# Resolution logic:
#   deepset and jasperls share identical parquet files. After exact dedup,
#   their shared rows are tagged "deepset|jasperls". We attribute these to
#   "deepset" in the figure (primary, most-cited source) with a footnote.
#   jasperls itself therefore contributes 0 unique rows post-dedup.
#
# All other counts are read directly from the deduped CSV grouped by source.
# ─────────────────────────────────────────────────────────────────────────────

# Canonical display order for the figure (left to right, logical grouping)
SOURCE_ORDER = ['deepset', 'gandalf', 'hackaprompt', 'mosscap', 'safeguard', 'spml']

# Human-readable x-axis labels for the figure
SOURCE_LABELS = {
    'deepset':     'deepset\n(incl. jasperls)',
    'gandalf':     'gandalf',
    'hackaprompt': 'hackaprompt',
    'mosscap':     'mosscap',
    'safeguard':   'safeguard',
    'spml':        'spml',
}

# Full HuggingFace IDs for the report table
SOURCE_HF_IDS = {
    'deepset':     'deepset/prompt-injections',
    'mosscap':     'Lakera/mosscap_prompt_injection',
    'gandalf':     'Lakera/gandalf_ignore_instructions',
    'hackaprompt': 'hackaprompt/hackaprompt-dataset',
    'spml':        'reshabhs/SPML_Chatbot_Prompt_Injection',
    'safeguard':   'xTRam1/safe-guard-prompt-injection',
    'jasperls':    'JasperLS/prompt-injections',
}


def compute_source_stats(df_raw: pd.DataFrame,
                         df_dedup: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-source raw and deduped counts.

    For the report table we show all 7 sources including jasperls.
    For the figure we use SOURCE_ORDER which excludes jasperls (zero rows).

    The deduped CSV source column contains merged tags like "deepset|jasperls".
    We use str.contains() matching so each source key picks up rows that
    contain its name, whether standalone or in a merged tag.
    """
    report_sources = ['deepset', 'mosscap', 'gandalf', 'hackaprompt',
                      'spml', 'safeguard', 'jasperls']
    rows = []
    for key in report_sources:
        raw_count   = df_raw[df_raw['source'].str.contains(key, na=False)].shape[0]
        dedup_count = df_dedup[df_dedup['source'].str.contains(key, na=False)].shape[0]
        rows.append({
            'source_key':   key,
            'hf_id':        SOURCE_HF_IDS[key],
            'raw_count':    raw_count,
            'dedup_count':  dedup_count,
            'removed':      raw_count - dedup_count,
        })
    return pd.DataFrame(rows)


def compute_figure_counts(df_dedup: pd.DataFrame) -> dict:
    """
    Compute per-source benign/injection counts for the figure.

    We resolve the merged source tags here:
      - "deepset|jasperls" and "deepset|jasperls|safeguard" rows are
        attributed to 'deepset' (primary source).
      - jasperls standalone rows are attributed to 'deepset' for the same reason.
      - safeguard standalone rows are attributed to 'safeguard'.
    """
    counts = {s: {'benign': 0, 'injection': 0} for s in SOURCE_ORDER}

    for _, row in df_dedup.iterrows():
        src   = str(row['source'])
        label = int(row['label'])
        key   = 'benign' if label == 0 else 'injection'

        # Resolve merged tags to a canonical single source
        if 'spml' in src:
            counts['spml'][key] += 1
        elif 'safeguard' in src and 'deepset' not in src and 'jasperls' not in src:
            counts['safeguard'][key] += 1
        elif 'deepset' in src or 'jasperls' in src:
            counts['deepset'][key] += 1
        elif 'gandalf' in src:
            counts['gandalf'][key] += 1
        elif 'hackaprompt' in src:
            counts['hackaprompt'][key] += 1
        elif 'mosscap' in src:
            counts['mosscap'][key] += 1
        elif 'safeguard' in src:
            counts['safeguard'][key] += 1

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure(df_dedup, df_train, df_val, df_test):
    BENIGN_COLOR  = '#2E86AB'
    INJECT_COLOR  = '#E84855'

    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         10,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'grid.color':        '#aaaaaa',
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')

    # ── LEFT PANEL: per-source grouped bars ──────────────────────────────────
    ax = axes[0]
    figure_counts = compute_figure_counts(df_dedup)

    benign_vals = [figure_counts[s]['benign']    for s in SOURCE_ORDER]
    inject_vals = [figure_counts[s]['injection'] for s in SOURCE_ORDER]
    totals      = [b + i for b, i in zip(benign_vals, inject_vals)]

    x         = np.arange(len(SOURCE_ORDER))
    bar_width = 0.36

    bars_b = ax.bar(x - bar_width / 2, benign_vals, bar_width,
                    color=BENIGN_COLOR, alpha=0.88,
                    edgecolor='white', linewidth=0.6)
    bars_i = ax.bar(x + bar_width / 2, inject_vals, bar_width,
                    color=INJECT_COLOR, alpha=0.88,
                    edgecolor='white', linewidth=0.6)

    # Total annotation above each source group
    y_max = max(totals) if max(totals) > 0 else 1
    for idx, (b, i, tot) in enumerate(zip(benign_vals, inject_vals, totals)):
        if tot > 0:
            ax.text(x[idx], max(b, i) + y_max * 0.02,
                    f'{tot:,}',
                    ha='center', va='bottom',
                    fontsize=8.5, color='#333', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SOURCE_LABELS[s] for s in SOURCE_ORDER],
        fontsize=9, rotation=20, ha='right', rotation_mode='anchor'
    )
    ax.set_ylabel('Number of samples', fontsize=10, labelpad=8)
    ax.set_title('(a) Samples per source after deduplication',
                 fontsize=11, pad=14, loc='left', fontweight='normal')
    ax.set_xlim(-0.6, len(SOURCE_ORDER) - 0.4)
    ax.set_ylim(0, y_max * 1.22)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.tick_params(axis='x', length=0)

    # Footnote explaining the deepset/jasperls merge
    ax.annotate(
        '* deepset/prompt-injections and JasperLS/prompt-injections share identical samples.\n'
        '  Duplicates removed by deduplication pipeline; remaining rows attributed to deepset.',
        xy=(0, -0.27), xycoords='axes fraction',
        fontsize=7.5, color='#666666', va='top'
    )

    # ── RIGHT PANEL: per-split grouped bars ───────────────────────────────────
    ax2 = axes[1]

    split_data = {
        'Train\n(n=12,504)':     {'benign': (df_train.label == 0).sum(),
                                   'injection': (df_train.label == 1).sum()},
        'Validation\n(n=2,680)': {'benign': (df_val.label   == 0).sum(),
                                   'injection': (df_val.label   == 1).sum()},
        'Test\n(n=2,680)':       {'benign': (df_test.label  == 0).sum(),
                                   'injection': (df_test.label  == 1).sum()},
    }

    split_labels  = list(split_data.keys())
    b_vals2 = [split_data[s]['benign']    for s in split_labels]
    i_vals2 = [split_data[s]['injection'] for s in split_labels]
    totals2 = [b + i for b, i in zip(b_vals2, i_vals2)]

    x2 = np.arange(len(split_labels))

    bars_b2 = ax2.bar(x2 - bar_width / 2, b_vals2, bar_width,
                      color=BENIGN_COLOR, alpha=0.88,
                      edgecolor='white', linewidth=0.6)
    bars_i2 = ax2.bar(x2 + bar_width / 2, i_vals2, bar_width,
                      color=INJECT_COLOR, alpha=0.88,
                      edgecolor='white', linewidth=0.6)

    # Percentage labels inside bars
    for bar, val, tot in zip(bars_b2, b_vals2, totals2):
        pct = val / tot * 100 if tot > 0 else 0
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() / 2,
                 f'{pct:.1f}%',
                 ha='center', va='center',
                 fontsize=9, color='white', fontweight='bold')

    for bar, val, tot in zip(bars_i2, i_vals2, totals2):
        pct = val / tot * 100 if tot > 0 else 0
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() / 2,
                 f'{pct:.1f}%',
                 ha='center', va='center',
                 fontsize=9, color='white', fontweight='bold')

    # Sample-count annotations above each split group
    y_max2 = max(max(b_vals2), max(i_vals2))
    for idx, (b, i) in enumerate(zip(b_vals2, i_vals2)):
        ax2.text(x2[idx] - bar_width / 2, b + y_max2 * 0.02,
                 f'{b:,}', ha='center', va='bottom',
                 fontsize=8, color=BENIGN_COLOR, fontweight='bold')
        ax2.text(x2[idx] + bar_width / 2, i + y_max2 * 0.02,
                 f'{i:,}', ha='center', va='bottom',
                 fontsize=8, color=INJECT_COLOR, fontweight='bold')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(split_labels, fontsize=9.5)
    ax2.set_ylabel('Number of samples', fontsize=10, labelpad=8)
    ax2.set_title('(b) Class distribution per split (stratified, seed = 42)',
                  fontsize=11, pad=14, loc='left', fontweight='normal')
    ax2.set_ylim(0, y_max2 * 1.22)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax2.tick_params(axis='x', length=0)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=BENIGN_COLOR, alpha=0.88, label='Benign (label = 0)'),
        mpatches.Patch(color=INJECT_COLOR, alpha=0.88, label='Injection (label = 1)'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=2,
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.suptitle(
        'Fig. 1 — SecureRAG Classifier Corpus: Source Composition and Split Statistics\n'
        'Total: 17,864 samples | 58.5% benign | 41.5% injection | 7 sources | seed = 42',
        fontsize=10.5,
        y=1.09,
        color='#333333',
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out_path = os.path.join(REPORT_DIR, 'class_distribution.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Text report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_text_report(df_raw, df_dedup, df_train, df_val, df_test,
                         source_stats: pd.DataFrame):
    total_raw   = df_raw.shape[0]
    total_dedup = df_dedup.shape[0]
    total_removed = total_raw - total_dedup

    n_train = len(df_train)
    n_val   = len(df_val)
    n_test  = len(df_test)

    def pct(n, total): return f'{n / total * 100:.1f}%'

    lines = [
        '=' * 72,
        'SECURERAG — DATASET CURATION REPORT',
        'Phase 1A: Classifier Training Dataset',
        '=' * 72,
        '',
        'TABLE 1: Source Dataset Composition',
        '-' * 72,
        f'{"Source":<14} {"HuggingFace ID":<43} {"Raw":>6} {"Dedup":>6} {"Removed":>8}',
        '-' * 72,
    ]

    for _, r in source_stats.iterrows():
        lines.append(
            f'{r["source_key"]:<14} {r["hf_id"]:<43} '
            f'{r["raw_count"]:>6,} {r["dedup_count"]:>6,} {r["removed"]:>8,}'
        )

    lines += [
        '-' * 72,
        f'{"TOTAL":<14} {"":<43} {total_raw:>6,} {total_dedup:>6,} {total_removed:>8,}',
        '',
        'NOTE: deepset/prompt-injections and JasperLS/prompt-injections share',
        '      identical samples. The dedup count for each reflects rows uniquely',
        '      attributable to that source after cross-source deduplication.',
        '      Shared rows appear once in the corpus, tagged "deepset|jasperls".',
        '',
        'Deduplication methodology:',
        '  Stage 1 — Exact dedup : SHA-256 hash of lowercased, stripped text.',
        '             Label conflicts resolved by keeping injection label (safer).',
        '  Stage 2 — Near-dedup  : Cosine similarity > 0.95 on all-MiniLM-L6-v2',
        '             embeddings (384-dim, L2-normalised). Later-encountered',
        '             sample removed.',
        '',
        'TABLE 2: Final Split Statistics',
        '-' * 72,
        f'{"Split":<22} {"Total":>8} {"Benign (0)":>12} {"Injection (1)":>14}'
        f' {"% Benign":>10} {"% Injection":>12}',
        '-' * 72,
    ]

    for name, df_s in [('Train', df_train),
                       ('Validation', df_val),
                       ('Test (FROZEN)', df_test)]:
        n   = len(df_s)
        b   = (df_s.label == 0).sum()
        inj = (df_s.label == 1).sum()
        lines.append(
            f'{name:<22} {n:>8,} {b:>12,} {inj:>14,}'
            f' {pct(b, n):>10} {pct(inj, n):>12}'
        )

    lines += [
        '-' * 72,
        '',
        'Split methodology:',
        '  Stratified random split preserving class ratio across all three splits.',
        '  Proportions: 70% train / 15% validation / 15% test.',
        '  Random seed: 42 (fixed for full reproducibility).',
        '  Test split is FROZEN — used only for final paper evaluation.',
        '  Never consulted during classifier training or threshold selection.',
        '',
        'TABLE 3: Class Distribution Summary',
        '-' * 72,
        f'  Raw merged  : {total_raw:,} total '
        f'| {(df_raw.label==0).sum():,} benign '
        f'| {(df_raw.label==1).sum():,} injection',
        f'  After dedup : {total_dedup:,} total '
        f'| {(df_dedup.label==0).sum():,} benign '
        f'| {(df_dedup.label==1).sum():,} injection',
        f'  Class ratio : {pct((df_dedup.label==0).sum(), total_dedup)} benign '
        f'/ {pct((df_dedup.label==1).sum(), total_dedup)} injection',
        '',
        'SAMPLING DECISIONS (documented for reproducibility):',
        '  mosscap    : 3,000 sampled from 277,148 raw rows (seed=42).',
        '               All rows are injection attempts (DEF CON 31 CTF).',
        '  hackaprompt: 3,000 sampled from ~298,783 filtered rows (seed=42).',
        '               Levels 8-10 excluded due to documented label noise.',
        '               All rows are injection attempts (HackAPrompt competition).',
        '  All other sources included in full after quality filtering.',
        '',
        '=' * 72,
    ]

    text = '\n'.join(lines)
    path = os.path.join(REPORT_DIR, 'dataset_report.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(text)
    print(f'\nReport saved: {path}')
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f'Loading datasets...')
    df_raw   = pd.read_csv(RAW_PATH)
    df_dedup = pd.read_csv(DEDUP_PATH)
    df_train = pd.read_csv(TRAIN_PATH)
    df_val   = pd.read_csv(VAL_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    print(f'  Raw   : {len(df_raw):,} rows')
    print(f'  Dedup : {len(df_dedup):,} rows')
    print(f'  Train : {len(df_train):,} | Val : {len(df_val):,} | Test : {len(df_test):,}')

    source_stats = compute_source_stats(df_raw, df_dedup)
    source_stats.to_csv(os.path.join(REPORT_DIR, 'dataset_stats.csv'), index=False)

    generate_text_report(df_raw, df_dedup, df_train, df_val, df_test, source_stats)
    generate_figure(df_dedup, df_train, df_val, df_test)

    print('\nAll outputs saved to:', REPORT_DIR)


if __name__ == '__main__':
    main()