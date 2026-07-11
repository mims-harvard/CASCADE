#!/usr/bin/env python3
"""
UMAP visualization of LUCA disease-context embeddings, coloured by disease,
cell_type, and tissue (one figure per variable).
"""
import colorsys
import gc
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

COLOR_VARS = ['disease', 'cell_type', 'tissue']
N_SUBSAMPLE = 50_000
RANDOM_SEED = 42
UMAP_PARAMS = dict(n_neighbors=200, min_dist=0.5, metric='cosine')

PT_SIZE = 2.0
ALPHA = 0.55
DPI = 200

PALETTE_40 = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
    '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
    '#1170AA', '#FC7D0B', '#A3ACB9', '#57606C', '#5FA2CE',
    '#C85200', '#7B848F', '#A3CEC9', '#D4A6C8', '#6E9A50',
    '#B6992D', '#86BCB6', '#D37295', '#499894', '#FABFD2',
    '#8CD17D', '#F1CE63', '#79706E', '#BCBD22', '#17BECF',
    '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
    '#C49C94', '#F7B6D2', '#C7C7C7', '#DBDB8D', '#9EDAE5',
]


def get_palette(labels):
    cats = sorted(set(str(l) for l in labels if str(l) not in ('nan', 'N/A', '')))
    n = len(cats)
    if n <= len(PALETTE_40):
        colours = PALETTE_40[:n]
    else:
        colours = []
        for i in range(n):
            r, g, b = colorsys.hls_to_rgb(i / n, 0.52, 0.72)
            colours.append('#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)))
    return {c: colours[i] for i, c in enumerate(cats)}


def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#f7f7f7')


def draw_scatter(ax, coords, labels_raw, palette):
    labels = np.array([str(l) for l in labels_raw])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(coords))
    cols = np.array([palette.get(labels[i], '#bbbbbb') for i in idx])
    ax.scatter(coords[idx, 0], coords[idx, 1], c=cols, s=PT_SIZE, alpha=ALPHA, linewidths=0, rasterized=True)


def add_legend(ax, palette, title='', max_cats=40):
    cats = sorted(palette.keys())[:max_cats]
    extra = len(palette) - max_cats
    handles = [mpatches.Patch(facecolor=palette[c], edgecolor='none', label=c) for c in cats]
    if extra > 0:
        handles.append(mpatches.Patch(facecolor='none', edgecolor='none', label=f'... +{extra} more'))
    ax.legend(handles=handles, title=title, title_fontsize=6, fontsize=5.5, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0, frameon=True, framealpha=0.85,
              edgecolor='#cccccc', ncol=max(1, len(cats) // 20), handlelength=1.1, handleheight=1.0)


def load_subsample(path, n, seed):
    print(f'Loading {path} ...', flush=True)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    emb = data['embedding']
    emb = emb.cpu().numpy() if hasattr(emb, 'cpu') else np.asarray(emb)

    total = emb.shape[0]
    print(f'  {total:,} cells -> subsampling {n:,}', flush=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, min(n, total), replace=False))
    emb_sub = emb[idx].astype(np.float32)
    del emb
    gc.collect()

    meta = {v: np.array(data.get(v, ['N/A'] * total))[idx] for v in COLOR_VARS}
    return emb_sub, meta


def run_umap(emb):
    import umap as umap_lib
    print(f'  Running UMAP {UMAP_PARAMS} ...', flush=True)
    return umap_lib.UMAP(n_components=2, random_state=RANDOM_SEED, verbose=False, **UMAP_PARAMS).fit_transform(emb)


def make_figures(coords, meta, output_dir, title_prefix):
    for var in COLOR_VARS:
        labels = np.array([str(l) for l in meta[var]])
        palette = get_palette(labels)

        fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
        fig.patch.set_facecolor('white')
        fig.suptitle(f'{title_prefix}  |  coloured by {var}\n(n = {len(labels):,} cells)', fontsize=11, y=1.02)

        draw_scatter(ax, coords, labels, palette)
        style_ax(ax)
        ax.set_xlabel('UMAP 1', fontsize=8)
        ax.set_ylabel('UMAP 2', fontsize=8)
        add_legend(ax, palette, title=var)

        stem = Path(output_dir) / f'umap_{var}'
        for ext in ('pdf', 'png'):
            fig.savefig(f'{stem}.{ext}', dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {stem}.pdf', flush=True)


def main(embeddings_file, output_dir, title_prefix='LUCA disease-context embeddings'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    emb, meta = load_subsample(embeddings_file, N_SUBSAMPLE, RANDOM_SEED)
    coords = run_umap(emb)

    np.savez_compressed(str(Path(output_dir) / 'umap_coords.npz'), umap=coords, **meta)
    make_figures(coords, meta, output_dir, title_prefix)
    print(f'\nDone. Figures in {output_dir}', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings-file', type=str, required=True, help='Path to the chunked/merged embeddings pickle')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--title-prefix', type=str, default='LUCA disease-context embeddings')
    args = parser.parse_args()
    main(args.embeddings_file, args.output_dir, args.title_prefix)
