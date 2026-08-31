#Import libraries

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt, colors
import os
import seaborn as sns
from scipy.integrate import trapezoid
#Update parameters
params = {'legend.fontsize': 'x-large', 'axes.titlesize':'x-large',
         'axes.linewidth': 2, 'axes.labelsize' : 'x-large',
         'ytick.major.width': 2, 'ytick.minor.width': 2,
         'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}

plt.rcParams.update(params)



def load_fold_data_correlation(output_folder, model_name, trial_number, 
                                num_folds=10, file_prediction_name='predictions_LB20250527_BV20240725_data_for_AI_updated_train_fold{fold}_fix_GG_split_TIS_correlation_variance_split_by_TIS_.txt'):
    
    correlations = []
    for fold in range(num_folds):
        try:
            file_correlations = os.path.join(output_folder, f'trial_{trial_number}', 'output_figures', file_prediction_name.format(fold=fold))

            corr_df = pd.read_csv(file_correlations, sep='\t')
            correlations.append(corr_df)
        
        except:
            print(f'File not found for model {model_name}, trial {trial_number}, fold {fold}: {file_correlations}')
            continue

    correlations_df = pd.concat(correlations, ignore_index=True)
    #model_file = correlations_df.rename(columns={'r2': f'r2_{model_name}'})
    correlations_df['model'] = model_name
    print(f'Number of rows in correlations_df for model {model_name}: {len(correlations_df)}')
    return correlations_df


def plot_cutoff_variance_corr(merged_file, output_file=False, cutoffs = [0, 0.1, 0.5, 1, 2, 3, 5, 10]):
    #For each model plot how the correlation gets better at different cutoffs of variance explained by the TI
    #Make folder of file if it does not exist
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_models = merged_file.sort_values('model')['model'].unique()
    cutoff_data = []
    for cutoff in cutoffs:
        subset_data = merged_file[merged_file['var_measurement'] >= cutoff]
        print(f'Cutoff: {cutoff}, Number of data points: {len(subset_data)}')
        
        #Now compute average correlation for each model
        #Take the average of the r2 values for each model
        model_cutoff_data = {}
        for model in all_models:
            avg_r2 = subset_data[subset_data['model'] == model]['r2'].mean()
            model_cutoff_data[f'avg_r2_{model}'] = avg_r2
        
        cutoff_data.append({'cutoff': cutoff, **model_cutoff_data})

    cutoff_df = pd.DataFrame(cutoff_data)

    plt.figure(figsize=(7, 6))

    #Take cmap
    cmap = plt.get_cmap('tab20')
    auc_models = {}
    for i, model in enumerate(all_models):
        #compute AUC
        auc = np.trapezoid(cutoff_df[f'avg_r2_{model}'], cutoff_df['cutoff'])

        sns.lineplot(data=cutoff_df, x='cutoff', y=f'avg_r2_{model}', label=f'{model.replace("_", " ")} (AUC= {auc:.2f})', color=cmap(i), marker='o')

        auc_models[model] = auc


    plt.xlabel('Cutoff on Measurement Variance\nin sequences within same TIS')
    plt.ylabel('Average Pearson\'s r \npredictions vs measurements')
    #Put plot outside
    plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        #Get the full path of the output file
        full_output_path = os.path.abspath(output_file)
        print(f'Plot saved to {full_output_path}', flush=True)
    
    else:
        plt.show()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Plot correlation vs variance cutoff for different models.')
    parser.add_argument('--output_folder', type=str, required=True, nargs='+', help='Output path to the correlation files.')
    parser.add_argument('--model_names', type=str, required=True, nargs='+', help='List of model names corresponding to the output folders.')
    parser.add_argument('--trial_number', type=int, required=True, nargs='+', help='Trial number to consider in the folder.')
    parser.add_argument('--num_folds', type=int, default=10, help='Number of folds for cross-validation.')
    parser.add_argument('--output_file', type=str, default=False, help='Output file to save the plot. If not provided, the plot will be shown.')
    parser.add_argument('--cutoffs', type=float, nargs='+', default=[0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], help='List of variance cutoffs to consider.')

    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    #Make sure the output_folder and model_names and trial_number have the same length
    if not (len(args.output_folder) == len(args.model_names) == len(args.trial_number)):
        raise ValueError("The length of output_folder, model_names, and trial_number must be the same.")
    

    all_correlations = []
    for output_folder, model_name, trial_number in zip(args.output_folder, args.model_names, args.trial_number):
        print(f'Processing model: {model_name}, trial: {trial_number}, output folder: {output_folder}\n', flush=True)
        correlations_df = load_fold_data_correlation(output_folder, model_name, trial_number, num_folds=args.num_folds)
        print(f'Number of rows in correlations_df for model {model_name}: {len(correlations_df)}')
        all_correlations.append(correlations_df)
        print(f'-------------------------------------\n')

    merged_file = pd.concat(all_correlations, ignore_index=True)
    plot_cutoff_variance_corr(merged_file, cutoffs=args.cutoffs, output_file=args.output_file)

if __name__ == "__main__":
    main()