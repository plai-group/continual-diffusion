import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from improved_diffusion.script_util import str2bool
import seaborn as sns


NICKNAME_MAP = {
    "online": "Online",
    "er": "Replay (S)",
    "er-full": "Replay (L)",
    "offline": "Offline",
    "er-h": "Replay (S/H)",
    "er-h-full": "Replay (L/H)",
    "er-full-big": "Replay++ (L)",
    "offline-big": "Offline++",
}

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
                        help='Path prefix to the CSV files.')
    parser.add_argument('--plot_type', type=str, required=True, choices=["online_test", "timeline"],
                        help='What kind of metric to plot.')
    parser.add_argument('--output_path', type=str, default="plots/default/plot.png",
                        help='Directory to save the PNG file.')
    parser.add_argument("--exclude", type=str, nargs='+', default=[])
    parser.add_argument("--latex_fonts", type=str2bool, default=False)
    return parser.parse_args()

def plot_metrics(csv_dirs, out_path, plot_type, excluded_nicknames):
    """
    csv_paths should have format: <PATH>/<PREFIX>_<FRAME INDEX>/final.csv where <FRAME INDEX> is an integer.
    """
    csv_dirs = sorted(csv_dirs, key=lambda e: int(e.split('_')[-1]))  # sort CSVs by timestep index
    dfs = [pd.read_csv(os.path.join(path, 'final.csv')) for path in csv_dirs]
    metrics = [c for c in dfs[0].columns if c not in ['nickname', 'wandb'] and not c.endswith('-err')
                                            and not c.startswith('color-acc')]
    dfs = [df.rename(columns={metric: metric.rstrip('-') for metric in metrics}) for df in dfs]
    dfs = [df[~df['nickname'].isin(excluded_nicknames)] for df in dfs]
    for df in dfs:
        df['nickname'] = df['nickname'].map(NICKNAME_MAP).fillna(df['nickname'])

    if plot_type == "timeline":
        steps = [int(path.split('_')[-2]) for path in csv_dirs]
        xlabel_text = "Train Stream Frame Index"
    elif plot_type == "online_test":
        steps = [int(path.split('_')[-1]) for path in csv_dirs]
        xlabel_text = "\# Training Iterations"
    else:
        raise Exception(f"Unsupported Plot Type: {plot_type}")

    # supplementary_dfs = [pd.read_csv(os.path.join(path, 'summary.csv')) for path in csv_dirs]
    # raw_columns = {metric: [c for c in supplementary_dfs[0].columns if c.startswith(metric)] for metric in metrics}
    fig, ax = plt.subplots(1, len(metrics), sharex=True, figsize=(18,4))
    unique_nicknames = dfs[0]['nickname'].unique()
    palette = sns.color_palette("Set2", len(unique_nicknames))
    color_map = {nickname: palette[i] for i, nickname in enumerate(unique_nicknames)}

    for mid, metric in enumerate(metrics):
        metric = metric.rstrip('-')
        all_metric_info = dfs[0][['nickname', metric]].rename(columns={metric: f"{metric}_{steps[0]}"})
        for step, df in zip(steps[1:], dfs[1:]):
            metric_info = df[['nickname', metric]].rename(columns={metric: f"{metric}_{step}"})
            all_metric_info = all_metric_info.merge(metric_info, on='nickname')
        for nickname in all_metric_info['nickname']:
            x_axis = [i for i in range(len(steps))]
            y_axis = all_metric_info[all_metric_info['nickname']==nickname].drop(columns='nickname').values.flatten().tolist()

            # Get error values for each step for this metric and nickname
            error_values = [dfs[i].loc[dfs[i]['nickname'] == nickname, metric + '--err'].values[0] for i in range(len(steps))]
            if sum(error_values) == sum(error_values):  # Add error bars if information is available
                ax[mid].errorbar(x_axis, y_axis, yerr=error_values, label=nickname, capsize=3,
                                 elinewidth=0.5, color=color_map[nickname])

            ax[mid].title.set_text(metric)
            ax[mid].plot(x_axis, y_axis, label=nickname, color=color_map[nickname])
            ax[mid].set_xticks(x_axis)
            ax[mid].set_xticklabels(steps)
            ax[mid].set_xlabel(xlabel_text)

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
    plot_metrics(csv_dirs, args.output_path, args.plot_type, args.exclude)

if __name__ == '__main__':
    main()
