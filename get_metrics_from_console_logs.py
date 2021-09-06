import os 
import re
import argparse
import numpy as np
import pandas as pd
import logging
import visualize_metrics as vis

# constants for plot labeling
UNIT_PER_METRIC = {
    "env_steps": "Number of Steps",
    "train_steps": "Number of Steps",
    "collect_time": "Time in Seconds",
    "train_time": "Time in Seconds",
    "eval_time": "Time in Seconds",
    "eval_avg_return": "Average Evaluation Return"
}

# constants for extracting metrics
# structured the following way: 
# <name_of_metric>: (<regex_to_extract_metric>, <datatype_as_a_string (optional)>)
TRAIN_METRICS_REGEX = {
    "iteration": ("^ ITERATION (.*)$", "int"),
    "env_steps": ("^# Env\. Steps:   (.*)$", "int"),
    "train_steps": ("^# Train Steps:  (.*)$", "int"),
    "collect_time": ("^# Collect time: \[(.*)\]s$", "float"), 
    "train_time": ("^# Train time:   \[(.*)\]s$", "float")
}

EVAL_METRICS_REGEX = {
    "eval_time": ("^# Eval time: \[(.*)\]s$", "float"),
    "eval_avg_return": ("^# Eval average return: (.*)$", "float")
}

class MetricsExtractor():
    def __init__(self, args):
        """
        Read and save the flags parsed on the console.
        """
        self.file = args.file
        self.debug = args.debug
        self.is_print = args.is_print
        self.start_iteration = args.start_iter
        self.eval_start_iter = args.eval_start
        self.eval_interval = args.eval_interval
        self.is_show = args.is_show
        self.write_dir = args.write_dir

        self.train_metrics = {}
        self.eval_metrics = {}

    def cast_to(self, value, type):
        """
        Takes in a value and a type. It converts the value to that specific datatype.

        Note that type is expected to be a string not a python native datatype. So don't use 
        int, instead use 'int' as a string.
        
        Param
        ---
        
        :value: any
        
        Some value that should be cast into another type
        
        :type: str
        
        The type you want to convert the value to. Use string format.
        Available options are: 'int', 'float', 'str', 'string'
        """
        if type == "int":
            try:
                res = int(value)
                return res
            except:
                logging.warn(f"Could not convert value {value} to int")
                return value
        elif type == "float":
            try:
                res = float(value)
                return res
            except:
                logging.warn(f"Could not convert value {value} to float")
                return value
        elif type == "str" or type == "string":
                return value
        else:
            logging.warn(f"Could not convert value {value}")
            return

    def get_regex_group(self, pattern, dtype="str"):
        """
        Takes an a string regex pattern and searches for the first regex group.
        
        First regex group means: (.*)
        
        or any specific literals that might be in the group.

        Here is an example regex:
        '^Here is the value: (.*)$'

        This regex searches for a line that begins with the literals 'Here is the value: '
        and extracts the literals that follow until the end of the line.

        Param
        ---
        
        :pattern: str
        
        Regex pattern as a string

        :dtype: str
        
        The type you want to convert the value to. Use string format.
        Available options are: 'int', 'float', 'str', 'string'
        """
        results = []

        # patterns
        pat = re.compile(pattern)

        print(f"reading file {self.file}...")
        with open(self.file, 'r') as f:
            for line in f:
                #print(line)
                try:
                    # find the pattern in the line
                    match = pat.findall(line)
                    if match != []:
                        # It always returns a list of groups (e.g. "(.*)") 
                        # but there is only one group, therefore [0]
                        hit = match[0]
                        hit = self.cast_to(hit, dtype)
                        results.append(hit)
                    else:
                        if self.debug:
                            print(f"Did not find pattern: {pattern}")
                        logging.debug(f"Did not find pattern: {pattern}")
                except:
                    if self.debug:
                        print(f"Error in regex solution or conversion of regex result")
                    logging.debug(f"Error in regex solution or conversion of regex result")

        return results

    # def add_regex_to_collection(self, eval_or_train_metric, metric_name, pattern, dtype="str"):
    #     """
    #     Collects multiple regex patterns, the name of the metric they belong to and if it is an eval metric or a train metric.
    #     The collected data can then be used to read each line of a file only once and comparing each line with every regex.
    #     This is a better approach then reading the file anew for every regex comparison.
    #     """
    
    def extract(self):
        # Regex we want to collect into a list
        
        for metric in TRAIN_METRICS_REGEX.keys():
            self.train_metrics[metric] = self.get_regex_group(*TRAIN_METRICS_REGEX[metric])

        # Still need to collect the correct iteration where the eval metrics were extracted
        #self.eval_metrics["iteration"] = self.get_regex_group() wouldn't work because its not clear which iteration belongs to each eval step
        # workaround manually provide the eval start iteration and the iteration steps between evaluations
        # use the last element in train metrics iteration as the end of the evaluations
        # NOTE: this only works if the metrics are extracted in sequence it doen't work if they are extracted in parallel
        # This cannot be extracted with a regex
        self.eval_metrics["iteration"] = list(range(self.eval_start_iter, self.train_metrics["iteration"][-1], self.eval_interval))
        for metric in EVAL_METRICS_REGEX.keys():
            self.eval_metrics[metric] = self.get_regex_group(*EVAL_METRICS_REGEX[metric])  


        if self.is_print:  
            # print the values in the dict
            logging.info("########## Training metrics ##########")
            print("########## Training metrics ##########")
            for key, value in self.train_metrics.items():
                print(f"\n{key}:\n{value}")  
                logging.info(f"\n{key}:\n{value}")
            
            logging.info("########## Eval metrics ##########")
            print("########## Eval metrics ##########")
            for key, value in self.eval_metrics.items():
                print(f"\n{key}:\n{value}")
                logging.info(f"\n{key}:\n{value}")
        
        # If visualizations should be shown or saved
        is_write_dir = self.write_dir not in ["", None]
        if self.is_show or is_write_dir:
            visualizer = vis.MetricsVisualizer()
            train, eval = self.metric_dfs()
            train_metrics = [m for m in self.train_metrics.keys() if m != "iteration"]
            eval_metrics = [m for m in self.eval_metrics.keys() if m != "iteration"]

            # visualize all metrics save and/or show the plots
            for metric in train_metrics:
                visualizer.add_metric_to_line_plot(train, metric)
                visualizer.show(self.is_show, self.write_dir, start_y_at=0, y_label=UNIT_PER_METRIC[metric])

            for metric in eval_metrics:
                visualizer.add_metric_to_line_plot(eval, metric)
                visualizer.show(self.is_show, self.write_dir, start_y_at=0, y_label=UNIT_PER_METRIC[metric])
                

    def metric_dfs(self):
        assert self.train_metrics != {}, "No training metrics have been collected yet"
        assert self.eval_metrics != {}, "There are training metrics, but no eval metrics have been collected yet"

        try:
            train_df = pd.DataFrame(data=self.train_metrics)
            eval_df = pd.DataFrame(data=self.eval_metrics)
            if self.is_print:
                print(train_df)
                print(eval_df)
                logging.info(train_df)
                logging.info(eval_df)
            return train_df, eval_df
        except:
            logging.error(
                """
                Dataframe of training metrics could not be created.
                This is probably due to incomplete information about an iteration in the beginning or the end of the file.
                Make sure there are only blocks that all belong to one iteration, like so:
                ------------------------------------------------------------------------
                ITERATION 123
                ------------------------------------------------------------------------
                # Episodes:     1234
                # Env. Steps:   1234
                # Train Steps:  1234

                # Collect time: [12.34]s
                # Train time:   [12.34]s
                # TOTAL:        [12.34]s
                """)
            print(
                """
                Dataframe of training metrics could not be created.
                This is probably due to incomplete information about an iteration in the beginning or the end of the file.
                Make sure there are only blocks that all belong to one iteration, like so:
                ------------------------------------------------------------------------
                ITERATION 123
                ------------------------------------------------------------------------
                # Episodes:     1234
                # Env. Steps:   1234
                # Train Steps:  1234

                # Collect time: [12.34]s
                # Train time:   [12.34]s
                # TOTAL:        [12.34]s
                """)
            return
        

    def get_metrics_dicts(self):
        return self.train_metrics, self.eval_metrics


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Flags
    parser.add_argument("-f", "--file", help="the log file from which the data is read", type=str)
    parser.add_argument("-d", "--debug", dest='debug', action='store_true', help="Set this option to get debugging information. Default: false")
    parser.set_defaults(debug=False)
    parser.add_argument("-p", "--print", dest='is_print', action='store_true', help="Set this flag if you want to get the metrics printed out to the console. Default: false")
    parser.set_defaults(is_print=False)
    parser.add_argument("-s", "--start_iter", default=0, type=int, help="The first iteration that is provided in the console output. Might be usefull if you didn't start a fresh training. Default: 0")
    parser.add_argument("-e", "--eval_start", type=int, default=-1, help="This is the first iteration in which an evaluation is performed. Default: -1 (before first training iteration)")
    parser.add_argument("-i", "--eval_interval", type=int, default=5, help="The iterations between one evaluation and the next one. Default: 5")
    parser.add_argument("-w","--write_dir", type=str, default="", help="If you set a directory here the ploted metrics are saved to that directory if it exists. Default: Nothing is saved")
    parser.add_argument("--show", dest='is_show', action='store_true', help="Set this flag if you want to get the metrics displayed in plots. Default: false")
    parser.set_defaults(show=False)
    args = parser.parse_args()

    metr_ex = MetricsExtractor(args)
    metr_ex.extract()

    