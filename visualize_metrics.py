import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class MetricsVisualizer():
    def __init__(self):
        self.current_plot = None
        self.current_y_max = None
        self.current_x_max = None
        self.current_x_min = None
        self.current_legend_labels = []

    def add_metric_to_line_plot(self, df, metric_name, legend_label=None):
        
        # use the provided df and take df["iterations"] as x and df[metric_name] as y
        self.current_plot = sns.lineplot(data=df, x="iteration", y=metric_name)
        
        # Save information for x and y axis scaling
        try:
            self.current_y_max = max(df[metric_name])
            # current_y_min is not needed, because y will start at 0 (if not specified otherwise with flags)
        except:
            print("Couldn't determine upper y-axis limit")
        try:
            self.current_x_min, self.current_x_max = min(df["iteration"]), max(df["iteration"])
        except:
            print("Couldn't determine x-axis limits")
            
        # if no legend label provided use the name of the metric
        # otherwise use the provided label
        if legend_label is None:
            self.current_legend_labels.append(metric_name)
        else:
            self.current_legend_labels.append(legend_label)

    def clean_state(self):
        # clean up state for next plot
        self.current_plot = None
        self.current_y_max = None
        self.current_legend_labels = []

    def show(self, x_label="Iteration Number", y_label="Values", start_y_at=None):
        # apply x- and y-axis labels
        self.current_plot.set(xlabel=x_label, ylabel=y_label)
        # start y-axis at 0 for example
        # end y-axis at 130% of the max value in the plot
        if self.current_y_max is not None:
            y_max = self.current_y_max*1.3
        plt.ylim([start_y_at, y_max])
        if self.current_x_min is not None and self.current_x_max is not None:
            # make sure the plot starts at the lowest possible coordinate
            plt.xlim([self.current_x_min, self.current_x_max])
        else:
            # default to 0 as first iteration and let matplotlib determine the upper limit
            plt.xlim([0, None])
        
        if len(self.current_legend_labels) == 1:
            # if there is only one metric plotted use subtitle instead of legend
            plt.title(self.current_legend_labels[0])
        elif len(self.current_legend_labels) > 1:
            # use the names of the metrics in the legend
            plt.legend(labels=self.current_legend_labels)
        else:
            print("There is no label for this plot")
            
        plt.show()
        self.clean_state()



