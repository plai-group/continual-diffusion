import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Plot metrics from CSV files.')
    parser.add_argument('--csv_paths', nargs='+', type=str, required=True,
                                       help='Paths to the CSV files. Must be given in the order of ascending training epochs.',)
    parser.add_argument('--nicknames', nargs='+', type=str, default=None,
                                       help='If specified, only plot runs with these nicknames.',)
    parser.add_argument('--out_path', type=str, help='Where to save the CSV file.', default="metrics_plot.png")
    return parser.parse_args()

def plot_metrics(csv_paths, out_path, nicknames):
    dfs = [pd.read_csv(path) for path in csv_paths]
    steps = [int(path.split('/')[-2].split('_')[-1][1:]) for path in csv_paths]
    metrics = [c for c in dfs[0].columns if c not in ['nickname', 'wandb']]
    fig, ax = plt.subplots(len(metrics), sharex=True)

    for mid, metric in enumerate(metrics):
        all_metric_info = dfs[0][['nickname', metric]].rename(columns={metric: f"{metric}_{steps[0]}"})
        for step, df in zip(steps[1:], dfs[1:]):
            metric_info = df[['nickname', metric]].rename(columns={metric: f"{metric}_{step}"})
            all_metric_info = all_metric_info.merge(metric_info, on='nickname')
        if nicknames is not None:
            all_metric_info = all_metric_info[all_metric_info['nickname'].isin(nicknames)]
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
    plot_metrics(args.csv_paths, args.out_path, args.nicknames)

if __name__ == '__main__':
    main()
