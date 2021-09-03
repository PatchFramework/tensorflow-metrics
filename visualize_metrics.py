import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class MetricsVisualizer():
    def __init__(self):
        self.current_plot = None
        self.current_legend_labels = []

    def add_metric_to_line_plot(self, df, metric_name, legend_label=None):
        
        # use the provided df and take df["iterations"] as x and df[metric_name] as y
        self.current_plot = sns.lineplot(data=df, x="iteration", y=metric_name)
        
        # if no legend label provided use the name of the metric
        # otherwise use the provided label
        if legend_label is None:
            self.current_legend_labels.append(metric_name)
        else:
            self.current_legend_labels.append(legend_label)

    def show(self, x_label="iteration number", y_label="values", start_y_at=None):
        # apply x- and y-axis labels
        self.current_plot.set(xlabel=x_label, ylabel=y_label)
        # start y-axis at 0
        plt.ylim(start_y_at, None)
        # use the names of the metrics in the legend
        plt.legend(labels=self.current_legend_labels)
        plt.show()


