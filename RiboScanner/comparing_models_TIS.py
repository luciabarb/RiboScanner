#Import libraries

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt, colors
import os
import seaborn as sns
from scipy.integrate import trapezoid
import datetime
#Update parameters
params = {'legend.fontsize': 'x-large', 'axes.titlesize':'x-large',
         'axes.linewidth': 2, 'axes.labelsize' : 'x-large',
         'ytick.major.width': 2, 'ytick.minor.width': 2,
         'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}

plt.rcParams.update(params)



def load_fold_data_correlation(output_folder, model_name, trial_number, 
                                num_folds=10, file_prediction_name='predictions_LB20250527_BV20240725_data_for_AI_updated_train_fold{fold}_fix_GG_split_TIS_correlation_variance_split_by_TIS_ length.txt'):
    
    correlations = []
    for fold in range(num_folds):
        try:
            file_correlations = os.path.join(output_folder, f'trial_{trial_number}', 'output_figures', file_prediction_name.format(fold=fold))

            corr_df = pd.read_csv(file_correlations, sep='\t')
            correlations.append(corr_df)
        
        except:
            print(f'File not found for model {model_name}, trial {trial_number}, fold {fold}: {os.path.abspath(file_correlations)}')
            continue

    correlations_df = pd.concat(correlations, ignore_index=True)
    #model_file = correlations_df.rename(columns={'r2': f'r2_{model_name}'})
    correlations_df['model'] = model_name
    print(f'Number of rows for model {model_name}: {len(correlations_df)}')
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
        if 'TIS_ length' in subset_data.columns:
            print(f'Cutoff: {cutoff}, Number of rows: {len(subset_data)}, Number of unique TIS: {len(subset_data["TIS_ length"].unique())}')
        
        else:
            print(f'Cutoff: {cutoff}, Number of rows: {len(subset_data)}, Number of unique TIS: {len(subset_data["TIS_"].unique())}')
            
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
    
    return auc_models


def split_features_plot_heatmap(auc_models, output_file=False):

    model = list(auc_models.keys())
    #print(f'Model : {model}')

    padding_types = ['left', 'right', 'middle', 'random']
    gradient_clipping_types = ['5', '10', '20', '50']
    weighted_loss_types = ['03','07', '09', '2', '5']
    padding_info, gradient_info, weight_loss, adaptor_info = [], [], [], []

    for name in model:
        #Padding type
        if 'padding' in name or 'Padding' in name:
            if 'padding' in name: padding_type = name.split('padding_')[1].split('_')[0]
            else: padding_type = name.split('Padding_')[1].split('_')[0]
            #print(f'padding_type: {padding_type}')

            if padding_type in padding_types: padding_info.append(padding_type.capitalize())
            else: padding_info.append('Right')
        else: padding_info.append('Right')

        if 'gradient_clipping' in name:
            gradient_type = name.split('gradient_clipping_')[1].split('_')[0]
            if gradient_type in gradient_clipping_types: gradient_info.append(int(gradient_type))
            else: gradient_info.append('')
        
        else: gradient_info.append('no gradient\nclipping')
        
        if 'multitask' in name:
            if 'weighted_loss_' in name:
                weight_type = (name.split('weighted_loss_')[1].split('_')[0])
                if weight_type in weighted_loss_types: weight_loss.append(float(weight_type.replace('0', '0.')))
            else: weight_loss.append(1.0)

        else: weight_loss.append('no_multitask')
        
        if 'adaptors' in name:
            #print(f'Name: {name}')
            if 'seq_no_adaptors' in name:
                adaptor_info.append('No\nadaptors\nsequence')
            elif 'no_adaptors' in name:
                adaptor_info.append('No\nadaptors')
            elif 'seq_adaptors' in name:
                adaptor_info.append('Sequence\nadaptors')
            else:
                adaptor_info.append('Adaptors')
        else:
            adaptor_info.append('')
        
        # Feature 3: ['Adaptors', 'Adaptors', 'Adaptors', 'No adaptors', 'No adaptors', 'Adaptors', 'No adaptors', 'No adaptors', 'Adaptors', 'No adaptors', 'No adaptors', 'Adaptors', 'No adaptors', 'No adaptors']

    list_features = [padding_info, gradient_info, weight_loss, adaptor_info]

    for i, feature_1 in enumerate(list_features):
        for i2, feature_2 in enumerate(list_features):
            if (i2 > i) and (len(feature_1) != 0) and (len(feature_2) != 0) and (len(set(feature_1)) > 1) and (len(set(feature_2)) > 1):
                #print(f'Feature {i}: {feature_1} \n Feature {i2}: {feature_2} \n\n')

                n_cols = len(set(feature_2))
                n_rows = len(set(feature_1))

                fig, ax = plt.subplots(figsize=(n_cols*2, n_rows*1.5), nrows=2, ncols=2, 
                                gridspec_kw={'height_ratios': [1, 4], 'width_ratios': [4, 1]})
                feature_1_name = ['padding', 'gradient_clipping', 'weighted_loss', 'adaptors'][i]
                feature_2_name = ['padding', 'gradient_clipping', 'weighted_loss', 'adaptors'][i2]

                df_heatmap = pd.DataFrame({'feature_1': feature_1, 'feature_2': feature_2, 'auc': list(auc_models.values())})
                df_heatmap = df_heatmap.groupby(['feature_1', 'feature_2']).mean().reset_index()
                df_heatmap = df_heatmap.pivot(index='feature_1', columns='feature_2', values='auc')
                vmin = df_heatmap.min().min()
                vmax = df_heatmap.max().max()

                g = sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap='copper_r',cbar=False, linewidths=0.5, ax=ax[1, 0],
                                vmin=vmin, vmax=vmax)
                
                ax[1,0].set_yticklabels(ax[1,0].get_yticklabels(), rotation=0)
                ax[1,0].set_xticklabels(ax[1,0].get_xticklabels(), rotation=0)
                ax[1, 0].set_xlabel(f"{feature_2_name.capitalize().replace('_', ' ')} \n")
                ax[1, 0].set_ylabel(f"{feature_1_name.capitalize().replace('_', ' ')} \n")

                #Create cbar on the right of the heatmap
                cbar_ax = fig.add_axes([1, 0.13, 0.06, 0.5])
                norm = colors.Normalize(vmin=df_heatmap.min().min(), vmax=df_heatmap.max().max())
                sm = plt.cm.ScalarMappable(cmap='copper_r', norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cbar_ax)
                cbar.set_label('AUC', rotation=270, labelpad=25)


                #Do average of the AUC values for each feature_1, and then for feature_2 and put them on top of the heatmap and on the right of the heatmap
                avg_feature_1 = df_heatmap.mean(axis=1)
                avg_feature_2 = df_heatmap.mean(axis=0)
                sns.heatmap(avg_feature_2.to_frame().T, annot=True, fmt=".2f", cmap='copper_r', cbar=False, linewidths=0.5, ax=ax[0, 0], vmin=vmin, vmax=vmax)
                #Remove the tick labels from the top heatmap
                ax[0, 0].set_xticklabels([], rotation=45, ha='right')
                ax[0, 0].set_yticklabels([], rotation=0)
                ax[0, 0].set_ylabel('')
                ax[0, 0].set_xticks([])
                ax[0, 0].set_xlabel('')
                ax[0, 0].set_yticks([])
                sns.heatmap(avg_feature_1.to_frame(), annot=True, fmt=".2f", cmap='copper_r', cbar=False, linewidths=0.5, ax=ax[1, 1], vmin=vmin, vmax=vmax)
                ax[1, 1].set_xticklabels([], rotation=45, ha='right')
                ax[1, 1].set_yticklabels([], rotation=0)
                ax[1, 1].set_ylabel('')
                ax[1, 1].set_xlabel('')
                ax[1, 1].set_xticks([])
                ax[1, 1].set_yticks([])
                ax[0,1].axis('off')


                
                if output_file:
                    heatmap_output_file = os.path.splitext(output_file)[0] + f'_{feature_1_name}_{feature_2_name}.pdf'
                    plt.savefig(heatmap_output_file, bbox_inches='tight', dpi=300)
                    print(f'Heatmap saved to {os.path.abspath(heatmap_output_file)}', flush=True)
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
    if not (len(args.model_names) == len(args.trial_number)):
        raise ValueError("The length of model_names, and trial_number must be the same.")
    
    if len(args.output_folder) == 1:
        args.output_folder = args.output_folder*len(args.model_names)
    
    elif len(args.output_folder) != len(args.model_names) :
        raise ValueError("The length of output_folder, model_names, and trial_number must be the same.")

    
    

    all_correlations = []
    for output_folder, model_name, trial_number in zip(args.output_folder, args.model_names, args.trial_number):
        print(f'Processing model: {model_name}, trial: {trial_number}, output folder: {output_folder}\n', flush=True)
        correlations_df = load_fold_data_correlation(output_folder, model_name, trial_number, num_folds=args.num_folds)
        print(f'Number of rows in correlations_df for model {model_name}: {len(correlations_df)}')
        all_correlations.append(correlations_df)
        print(f'-------------------------------------\n')

    merged_file = pd.concat(all_correlations, ignore_index=True)
    auc_models = plot_cutoff_variance_corr(merged_file, cutoffs=args.cutoffs, output_file=args.output_file)

    output_heatmap = os.path.splitext(args.output_file)[0] + '_heatmap.pdf' if args.output_file else False

    split_features_plot_heatmap(auc_models, output_heatmap)
    
    #In output_file change extension to .txt and save the arguments used to run the script in that file
    if args.output_file:
        output_file_txt = os.path.splitext(args.output_file)[0] + '.txt'
        with open(output_file_txt, 'w') as f:
            f.write(f'Arguments used to run the script on {datetime.datetime.now()}:\n')
            f.write(f'Output folders: {args.output_folder}\n')
            f.write(f'Model names: {args.model_names}\n')
            f.write(f'Trial numbers: {args.trial_number}\n')
            f.write(f'Number of folds: {args.num_folds}\n')
            f.write(f'Cutoffs: {args.cutoffs}\n')
            f.write(f'Output file: {args.output_file}\n')

if __name__ == "__main__":
    main()