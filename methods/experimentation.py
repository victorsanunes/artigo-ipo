from preprocessing import preprocess
import pandas as pd

def run_optimization(df, target_columns, optimization_method, **kwargs):
    X = df.drop(columns = target_columns)
    Y = df[target_columns]
#     norm_X, norm_Y = preprocess(X, Y)
#     output = optimization_method(norm_X, norm_Y, **kwargs)
    output = optimization_method(X, Y, **kwargs)
    return output

def create_results_dataframe(experiment_output):
    results_df = pd.DataFrame()

    results_df['dataset'] = experiment_output.keys()
    results_df['error'] = [experiment_output[dataset_filename].error for dataset_filename in experiment_output.keys()]
    results_df['execution_time'] = [experiment_output[dataset_filename].execution_time for dataset_filename in experiment_output.keys()]
    results_df['method'] = [experiment_output[dataset_filename].method_name for dataset_filename in experiment_output.keys()]
    results_df['n_tries'] = [experiment_output[dataset_filename].n_tries for dataset_filename in experiment_output.keys()]
    results_df['search_space'] = [experiment_output[dataset_filename].search_space for dataset_filename in experiment_output.keys()]

    return results_df