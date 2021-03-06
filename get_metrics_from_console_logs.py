import os 
import re
import argparse
import numpy as np
import pandas as pd
import logging
import visualize_metrics as vis
import copy

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
# NOTE: The parantheses are required
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

# This regex will be removed from the input stream
# To disable this feature just provide an empty string
COLOR_CODE_REGEX = "\x1b\[[0-9;]*[a-zA-Z]"

class MetricsExtractor():
    def __init__(self, args, use_file_idx):
        """
        Read and save the flags parsed on the console.
        """
        self.file = args.file[use_file_idx]
        self.is_multiple_files = len(args.file) > 1
        self.debug = args.debug
        self.is_print = args.is_print
        self.start_iteration = args.start_iter
        self.eval_start_iter = args.eval_start
        self.eval_interval = args.eval_interval
        self.is_show = args.is_show
        self.write_dir = args.write_dir
        self.mark = args.mark

        self.metrics = {
            "train": {},
            "eval": {}
        }

        # list of compiled regex patterns
        self.patterns = {
            "train": {},
            "eval": {}
        }
        self.metric_dtypes = {}

        self.color_regex = re.compile(COLOR_CODE_REGEX)
        # whether to clean the input stream or not
        self.is_not_clean = COLOR_CODE_REGEX == ""

        self.df = {
            "train": None,
            "eval": None
        }
        # holds the name of the metrics that should be visualized
        self.visualize = {
            "train": [],
            "eval": []
        }

    def clean_stream_from_color_coding(self, line):
        """
        Removes any color coding from the input stream of a file.
        The file itself is not modified.
        """
        if self.is_not_clean:
            return line
        else:
            cleaned = re.sub(self.color_regex, "", line)
            return cleaned

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

    def construct_metrics_dict(self, kind_of_metric, metric):
        """
        Create empty lists in the metrics dict, so that the values 
        for that metric can be appended in the next steps.
        """
        self.metrics[kind_of_metric][metric] = []

    def compile_regex(self, kind_of_metric, title, pattern):
        """
        Use the provided regex string pattern to compile it and save it in state.
        """
        pat = re.compile(pattern)
        self.patterns[kind_of_metric][title] = pat

    def get_regex_groups(self):
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
        print(f"reading file {self.file}...")
        with open(self.file, 'r') as f:

            for line in f:
                line = self.clean_stream_from_color_coding(line)

                for kind_of_metric, m_dict in self.patterns.items():
                    # kind_of_metric is "train" or "eval"
                    # m_dict is a dictionary that holds the metrics(key) and a list of their values(value)
                
                    # check the line for all patterns
                    for title, pattern in m_dict.items():
                        try:
                            # find the pattern in the line
                            match = pattern.findall(line)
                            # if there was a match (match not empty)
                            if match != []:
                                # It always returns a list of groups (e.g. "(.*)") 
                                # but there is only one group in each regex, therefore [0]
                                hit = match[0]
                                
                                # Read the respective data type or use str if it's not specified
                                if title in self.metric_dtypes.keys():
                                    d_type = self.metric_dtypes[title]
                                else:
                                    d_type = "str"
                                # convert the datatype
                                hit = self.cast_to(hit, d_type)

                                # Append the value to the metric to the list with the metric title
                                self.metrics[kind_of_metric][title].append(hit)
                            else:
                                if self.debug:
                                    print(f"Did not find pattern: {pattern}")
                                logging.debug(f"Did not find pattern: {pattern}")
                        except:
                            if self.debug:
                                print(f"Error in regex solution or conversion of regex result")
                            logging.debug(f"Error in regex solution or conversion of regex result")     

    # def add_regex_to_collection(self, eval_or_train_metric, metric_name, pattern, dtype="str"):
    #     """
    #     Collects multiple regex patterns, the name of the metric they belong to and if it is an eval metric or a train metric.
    #     The collected data can then be used to read each line of a file only once and comparing each line with every regex.
    #     This is a better approach then reading the file anew for every regex comparison.
    #     """
    
    def extract(self):
        # Regex we want to compile and collect into a list
        for metric, tuple in TRAIN_METRICS_REGEX.items():
            self.construct_metrics_dict("train", metric)
            # tuple is structured like this: (<regex_to_extract_metric>, <datatype_as_a_string (optional)>)
            self.compile_regex("train", metric, tuple[0])
            if len(tuple) >=2:
                # as the datatype is optional this might throw an error
                self.metric_dtypes[metric] = tuple[1]
            else:
                if self.debug:
                    print(f"No datatype provided for metric {metric}")
                logging.debug(f"No datatype provided for metric {metric}")

            # self.train_metrics[metric] = self.get_regex_group(*tuple)
        for metric, tuple in EVAL_METRICS_REGEX.items():
            self.construct_metrics_dict("eval", metric)
            # tuple is structured like this: (<regex_to_extract_metric>, <datatype_as_a_string (optional)>)
            self.compile_regex("eval", metric, tuple[0])
            if len(tuple) >=2:
                # as the datatype is optional this might throw an error
                self.metric_dtypes[metric] = tuple[1]
            else:
                if self.debug:
                    print(f"No datatype provided for metric {metric}")
                logging.debug(f"No datatype provided for metric {metric}")

        # Now check each line for that collection of patterns
        self.get_regex_groups()

        # Still need to collect the correct iteration where the eval metrics were extracted
        #self.eval_metrics["iteration"] = self.get_regex_group() wouldn't work because its not clear which iteration belongs to each eval step
        # workaround manually provide the eval start iteration and the iteration steps between evaluations
        # use the last element in train metrics iteration as the end of the evaluations
        # NOTE: this only works if the metrics are extracted in sequence it doen't work if they are extracted in parallel
        # Eval metrics iterations cannot be extracted with a regex
        try:
            amount_eval_time_metrics = len(self.metrics["eval"]["eval_time"])
            amount_eval_avg_return_metrics = len(self.metrics["eval"]["eval_avg_return"])
        except:
            logging.warning("There are no eval metrics")
        
        # If there are same amount of eval metrics, then create the same amount of iterations
        # Use user provided evaluation intervalls to determine which iteration the metrics belong to
        if amount_eval_avg_return_metrics == amount_eval_time_metrics:
            amount_eval_iterations = amount_eval_avg_return_metrics
            # e.g. start iteration is -1, 10 evaluations were conducted in an interval of 1 eval every 5 iterations
            # -1 + 10 * 5 = 49 <- this is the last iteration where eval takes place 
            last_eval_iteration = self.eval_start_iter + amount_eval_iterations * self.eval_interval
            self.metrics["eval"]["iteration"] = list(range(self.eval_start_iter, last_eval_iteration, self.eval_interval))
        else:
            logging.warning("The number of collected eval metrics is not identical, the eval dataframe cannot be created")

        if self.is_print:  
            # print the values in the dict
            for kind_of_metric, m_dict in self.metrics.items():
                logging.info(f"########## {kind_of_metric} metrics ##########")
                print(f"########## {kind_of_metric} metrics ##########")
                for key, value in m_dict.items():
                    print(f"\n{key}:\n{value}\nLength: {len(value)}")  
                    logging.info(f"\n{key}:\n{value}")
            
        
        # If visualizations should be shown or saved imediately
        is_write_dir = self.write_dir not in ["", None]
        if self.is_show or is_write_dir:
            self.df["train"], self.df["eval"] = self.metric_dfs()
            self.visualize["train"]  = [m for m in self.metrics["train"].keys() if m != "iteration"]
            self.visualize["eval"] = [m for m in self.metrics["eval"].keys() if m != "iteration"]
            
            # If there are multiple input logs, don't visualize everythin now
            # just save the data for visualization later
            if not self.is_multiple_files:
                visualizer = vis.MetricsVisualizer()

                # visualize all metrics save and/or show the plots
                for metric in self.visualize["train"]:
                    visualizer.add_metric_to_line_plot(self.df["train"], metric)
                    visualizer.show(self.is_show, self.write_dir, start_y_at=0, y_label=UNIT_PER_METRIC[metric], mark=self.mark)

                for metric in self.visualize["eval"]:
                    visualizer.add_metric_to_line_plot(self.df["eval"], metric)
                    visualizer.show(self.is_show, self.write_dir, start_y_at=0, y_label=UNIT_PER_METRIC[metric], mark=self.mark)
                

    def metric_dfs(self):
        assert self.metrics["train"] != {}, "No training metrics have been collected yet"
        assert self.metrics["eval"] != {}, "There are training metrics, but no eval metrics have been collected yet"

        try:
            logging.debug("Starting dataframe creation")
            if self.debug:
                print("Starting dataframe creation")
            train_df = pd.DataFrame(data=self.metrics["train"])
            eval_df = pd.DataFrame(data=self.metrics["eval"])
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

                OR

                Maybe you haven't set the --eval_start flag correctly
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
                
                OR

                Maybe you haven't set the --eval_start flag correctly
                """)
            return



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Flags
    # parser.add_argument("-f", "--file", help="the log file from which the data is read", type=str)
    
    parser.add_argument("-f", "--file", action="append", help="the log file from which the data is read.\
         You can use this flag multiple times, if you want to compare metrics from different runs.\
             This option may cause problems with runs that don't start at iteration 0. ", type=str)
    parser.add_argument("-l","--label", action="append", help="A label that will be used in the legend of a plot. \
        Use this flag once for each file that is provided with the -f/--file flag. Default: None")
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
    parser.add_argument("-m", "--mark", help="Mark a certain iteration with a red label. If labels overlap they are rotated. Deafult:None", type=int)
    args = parser.parse_args()

    try:
        # create one metrics extractor object per file provided
        extractors = [MetricsExtractor(copy.deepcopy(args), use_file_idx=idx) for idx in range(len(args.file))]
        print(extractors)
        for ex in extractors:
            # This will visualize and save plots if there is only one file provided
            # if there are multiple files it will just save train_df, eval_df and visualize in state
            # They can then be used to visualize the metrics from different runs in the same plot
            ex.extract()
        # visualize the same metrics of multiple runs in the same plot
        if len(args.file) > 1:
            visualizer = vis.MetricsVisualizer()

            # Get info from an extractor about the metrics that should be visualized
            for kind_of_metric, metric_list in extractors[0].visualize.items(): 
                # TODO: check if all extractors have that metric
                # TODO: add try except

                for metric in metric_list:
                    # Add the metric for each extracted file to the plot
                    for ex_id, ex in enumerate(extractors):
                        # plot the train of eval df of this specific extractor, with the name of the metric and 
                        # the label (-l) that was provided together with the respective -f flag

                        # TODO: add try/except here
                        
                        if args.debug:
                            print(f"Determining labels for extractor {ex_id}: {ex}")
                        logging.debug(f"Determining labels for extractor {ex_id}: {ex}")
                        # determine the label for this ploted extractor metric 
                        if args.label is not None and len(args.label) >= ex_id:
                            label = args.label[ex_id]
                        # otherwise use no labels
                        else:
                            label = None
                        
                        if args.debug:
                            print(f"Adding metric {metric} from extractor {ex_id}: {ex} with label {label} to the plot")
                        logging.debug(f"Adding metric {metric} from extractor {ex_id}: {ex} with label {label} to the plot")

                        visualizer.add_metric_to_line_plot(ex.df[kind_of_metric], metric, label)
                    
                    if args.debug:
                        print(f"visualizing plot for metric {metric}")
                    logging.debug(f"visualizing plot for metric {metric}")

                    # visualize this metric
                    visualizer.show(args.is_show, args.write_dir, y_label=UNIT_PER_METRIC[metric], start_y_at=0, mark=args.mark)

    except:
        print("Error: you need to provide at leased one console log file as input")
        logging.error("Error: you need to provide at leased one console log file as input")

    