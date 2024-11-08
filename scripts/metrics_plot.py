import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from improved_diffusion.script_util import str2bool


def setup_matplotlib():
    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
    plt.rcParams.update(nice_fonts)

def parse_args():
    parser = argparse.ArgumentParser(description='Plot metrics from CSV files.')
    parser.add_argument('--csv_path_prefix', type=str, required=True,
                        help='Path prefix to the CSV files.',)
    parser.add_argument('--output_path', type=str, help='Directory to save the PNG file.', default="plots/default/plot.png")
    parser.add_argument("--latex_fonts", type=str2bool, default=False)
    return parser.parse_args()

def plot_metrics(csv_dirs, out_path):
    """
    csv_paths should have format: <PATH>/<PREFIX>_<FRAME INDEX>/final.csv where <FRAME INDEX> is an integer.
    """
    csv_dirs = sorted(csv_dirs, key=lambda e: int(e.split('_')[-1]))  # sort CSVs by timestep index
    dfs = [pd.read_csv(os.path.join(path, 'final.csv')) for path in csv_dirs]
    steps = [int(path.split('_')[-1]) for path in csv_dirs]
    metrics = [c for c in dfs[0].columns if c not in ['nickname', 'wandb'] and not c.endswith('-err')]
    dfs = [df.rename(columns={metric: metric.rstrip('-') for metric in metrics}) for df in dfs]
    fig, ax = plt.subplots(1, len(metrics), sharex=True, figsize=(16,4))

    for mid, metric in enumerate(metrics):
        metric = metric.rstrip('-')
        all_metric_info = dfs[0][['nickname', metric]].rename(columns={metric: f"{metric}_{steps[0]}"})
        for step, df in zip(steps[1:], dfs[1:]):
            metric_info = df[['nickname', metric]].rename(columns={metric: f"{metric}_{step}"})
            all_metric_info = all_metric_info.merge(metric_info, on='nickname')
        for nickname in all_metric_info['nickname']:
            x_axis = [i for i in range(len(steps))]
            y_axis = all_metric_info[all_metric_info['nickname']==nickname].drop(columns='nickname').values.flatten().tolist()
            ax[mid].title.set_text(metric)
            ax[mid].plot(x_axis, y_axis, label=nickname)
            ax[mid].set_xticks(x_axis)
            ax[mid].set_xticklabels(steps)

    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path)

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.latex_fonts:
        setup_matplotlib()
    csv_dirs = glob.glob(args.csv_path_prefix + "*")
    plot_metrics(csv_dirs, args.output_path)

if __name__ == '__main__':
    main()
