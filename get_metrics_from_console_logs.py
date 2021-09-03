import os 
import re
import argparse
import numpy as np
import pandas as pd
import logging

class MetricsExtractor():
    def __init__(self, args):
        """
        Read and save the flags parsed on the console.
        """
        self.file = args.file
        self.debug = args.debug
        self.is_print = args.is_print
        self.start_iteration = args.start_iter
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
                print(f"Could not convert value {value} to int")
                return value
        elif type == "float":
            try:
                res = float(value)
                return res
            except:
                print(f"Could not convert value {value} to float")
                return value
        elif type == "str" or type == "string":
                return value
        else:
            print(f"Could not convert value {value}")
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
                except:
                    if self.debug:
                        print(f"Error in regex solution or conversion of regex result")
                    # pass

        return results
    
    def extract(self):
        # Regex we want to collect into a list
        
        # Still need to collect the correct iteration where the eval metrics were extracted
        #self.eval_metrics["iteration"] = self.get_regex_group()
        self.eval_metrics["eval_time"] = self.get_regex_group('^# Eval time: \[(.*)\]s$', "float")
        self.eval_metrics["eval_avg_return"] = self.get_regex_group('^# Eval average return: (.*)$', "float")
        
        self.train_metrics["iteration"] = self.get_regex_group('^ ITERATION (.*)$', "int")
        self.train_metrics["env_steps"] = self.get_regex_group('^# Env\. Steps:   (.*)$', "int")
        self.train_metrics["train_steps"] = self.get_regex_group('^# Train Steps:  (.*)$', "int")
        self.train_metrics["collect_time"] = self.get_regex_group('^# Collect time: \[(.*)\]s$', "float")
        self.train_metrics["train_time"] = self.get_regex_group('^# Train time:   \[(.*)\]s$', "float")

        if self.is_print:  
            # print the values in the dict
            print("########## Training metrics ##########")
            for key, value in self.train_metrics.items():
                print(f"\n{key}:\n{value}")  
            
            print("########## Eval metrics ##########")
            for key, value in self.eval_metrics.items():
                print(f"\n{key}:\n{value}")  

    def metrics_df(self):
        assert self.train_metrics != {}, "No training metrics have been collected yet"
        assert self.eval_metrics != {}, "There are training metrics, but no eval metrics have been collected yet"

        try:
            train_df = pd.DataFrame(data=self.train_metrics)
            eval_df = pd.DataFrame(data=self.eval_metrics)
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
        print(train_df)
        print(eval_df)

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
    args = parser.parse_args()

    metr_ex = MetricsExtractor(args)
    metr_ex.extract()
    metr_ex.metrics_df()

    #print(metr_ex.get_metrics_dict())
    