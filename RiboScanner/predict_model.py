import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt, colors

from .utils_model import load_model, getOneHot


#Update parameters
params = {'legend.fontsize': 'x-large', 'axes.titlesize':'x-large',
         'axes.linewidth': 2, 'axes.labelsize' : 'x-large',
         'ytick.major.width': 2, 'ytick.minor.width': 2,
         'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}

plt.rcParams.update(params)


def _load_external_predictor(model_path, model_type, device,
                              noderer_order='di', noderer_aug_col=None,
                              noderer_positions=None, label_scaler_path=None,
                              evo2_batch_size=32, evo2_regression_path=None):
    """
    Load a Noderer PWM or Evo2 predictor from .utils_riboscanner_extensions.
    Returns an object with a .predict(sequences, **kwargs) method.
    """
    from .utils_riboscanner_extensions import load_noderer_predictor, load_evo2_predictor

    if model_type == 'noderer':
        return load_noderer_predictor(
            coef_path=model_path,
            order=noderer_order,
            positions=noderer_positions,
            aug_col=noderer_aug_col,
            label_scaler_path=label_scaler_path,
        )
    elif model_type == 'evo2':
        return load_evo2_predictor(
            ckpt_path=model_path,
            device=device,
            evo2_regression_path=evo2_regression_path,
            batch_size=evo2_batch_size,
            label_scaler_path=label_scaler_path,
        )
    else:
        raise ValueError(f"Unknown external model_type: {model_type!r}")


# =============================================================================
# Core prediction functions
# =============================================================================


def predict_from_seq(models, seqs, L_max, padding='left', padding_value=0, batch_size=2000, 
                        variance_models = False, adaptors=False, model_type='MTtrans',
                        # Noderer-specific
                        noderer_order='di', noderer_aug_col=None, noderer_positions=None,
                        noderer_aug_indices=None,
                        # Evo2-specific
                        evo2_batch_size=32,
                        # Shared
                        label_scaler_path=None, device='cuda', padding_with_sequence=False):
    """
    This function will predict the output of a model(s) given a list of sequences.
    Args:
        models: (list) of models, should be the same model but different folds
        seqs: (list) of string sequences
        L_max: (int) Max length of sequences. Relevant for padding.
        padding: (str) Type of padding, only 3 possible (left, middle, right). (default: middle)
        padding_value: (int or float ) Value to pad with. (default: 0)
        variance_models: (bool) If true, store variance of the models and return it
    """
    # ── Noderer PWM ───────────────────────────────────────────────────────────
    if model_type == 'noderer':
        # models is already a loaded NodererPredictor
        predictor = models if not isinstance(models, list) else models[0]
        preds = predictor.predict(seqs, aug_indices=noderer_aug_indices)
        return preds.reshape(-1, 1)   # match MTtrans output shape (N, 1)

    # ── Evo2 ─────────────────────────────────────────────────────────────────
    if model_type == 'evo2':
        predictor = models if not isinstance(models, list) else models[0]
        preds = predictor.predict(seqs)
        return preds.reshape(-1, 1)

    #print(f'Models {models}  {type(models)}', flush=True)

    if type(models) != list: models = [models]
    if type(seqs) != list: seqs = [seqs]

    #print(f'Models {models} n_models {len(models)}', flush=True)

    #Split the sequences in batches
    batches = [seqs[i:i+batch_size] for i in range(0, len(seqs), batch_size)]

    #Loop over the batches
    predictions = []
    if variance_models: variances = []

    batches_loader = tqdm(batches, total = len(batches), ncols=80)
    for batch in batches_loader:
        pred_models = []
        if variance_models: var_models = []
        #Transform the sequences to one hot

        if model_type == 'GemoRNA':
            from .utils_external_models import prepare_input
            from .utils_external_models import five_prime_utr_vocab
            onehot = prepare_input(batch, vocab = five_prime_utr_vocab, pad_to=L_max)
            onehot = torch.tensor(onehot)
        
        else: 
            print(f' Padding sequences to length {L_max} with padding type {padding} and padding value {padding_value} padding_with_sequence {padding_with_sequence}', flush=True)
            onehot = getOneHot(batch, L_max, padding = padding, padding_value=padding_value, adaptors=adaptors,
                                        relative_to_start_codon = True if model_type == 'dense_layers' else False, padding_with_sequence=padding_with_sequence)
            #Transform to tensor and to gpus if available
            onehot = torch.tensor(np.float32(onehot))
            onehot = onehot.permute(0,2,1)
        if torch.cuda.is_available(): onehot = onehot.cuda()
        #Now loop over the models
        for i_model, model in enumerate(models):
            #print(f'Model {i_model}: {type(model)}', flush=True)
            model.eval()
            with torch.no_grad():
                outputs = model(onehot).cpu().detach().numpy()
                pred_models.append(outputs)
            
            #print the first item of each value of the list
        
        #Now average the predictions over the models
        predictions.append(np.mean(pred_models, axis=0))
        
        if variance_models: 
            variances.append(np.var(pred_models, axis=0))
    
    predictions = np.concatenate(predictions, axis=0)
    if variance_models: 
        variances = np.concatenate(variances, axis=0)
        return predictions, variances

    else: return predictions

def predict_from_fasta(input_file, models, L_max, output_file = False,
                       store_variance=False, padding='left', padding_value=0, batch_size=2000, adaptors=False, verbose=False, header_only=False,
                       model_type='MTtrans', padding_with_sequence=False):
    """
    This function will predict the output of a model(s) given a fasta file with sequences.
    Args:
        input_file: (str) path to the input fasta file
        models: (list) of models, should be the same model but different folds
        L_max: (int) Max length of sequences. Relevant for padding.
        output_file: (str or False) if False, it will return the dataframe with the predictions, if a string is given, it will save the dataframe to that path. (default: False)
    """

    #Create folder of output file if it doesn't exist
    if output_file is not False:
        output_folder = os.path.dirname(output_file)
        os.makedirs(output_folder, exist_ok=True)
        
    #Load the fasta file
    from Bio import SeqIO
    records = list(SeqIO.parse(input_file, "fasta"))
    seqs = [str(record.seq) for record in records]
    headers = [record.id for record in records] 

    # ── Load external predictors (noderer / evo2) ─────────────────────────────
    if model_type in ('noderer', 'evo2'):
        model_path = models[0] if isinstance(models, list) else models
        predictor  = _load_external_predictor(
            model_path, model_type, device,
            noderer_order=noderer_order, noderer_positions=noderer_positions,
            label_scaler_path=label_scaler_path,
            evo2_batch_size=evo2_batch_size,
            evo2_regression_path=evo2_regression_path,
        )
        predictions = predict_from_seq(
            predictor, seqs, L_max, model_type=model_type,
            evo2_batch_size=evo2_batch_size, device=device,
            padding_with_sequence=padding_with_sequence
        )

    else:
        #Load each model
        loaded_models_list = []
        for model in models:
            loaded_model = load_model(model, model=model_type, train=False, verbose=verbose, L_max=L_max)
            if torch.cuda.is_available(): loaded_model = loaded_model.cuda()
            loaded_model.eval()
            loaded_models_list.append(loaded_model)
        print(f'Number of models loaded: {len(loaded_models_list)}', flush=True)

        #Predict
        #Predict
        if store_variance: 
            predictions, variances = predict_from_seq(loaded_models_list, seqs, L_max, padding=padding, 
                                                    padding_value=padding_value, batch_size=batch_size, 
                                                    variance_models = store_variance, adaptors=adaptors, model_type=model_type, padding_with_sequence=padding_with_sequence)
        else: 
            predictions = predict_from_seq(loaded_models_list, seqs, L_max, padding=padding, padding_value=padding_value, 
                                        batch_size=batch_size, variance_models = store_variance, adaptors=adaptors, model_type=model_type, padding_with_sequence=padding_with_sequence)
    
    #Return a dataframe with the headers and the predictions
    if header_only:
        df = pd.DataFrame({'header': headers, 'predictions': predictions.flatten()})
    else: 
        df = pd.DataFrame({'header': headers, 'sequence': seqs, 'predictions': predictions.flatten()})
    
    if store_variance and 'variances' in dir():
        df['variance'] = variances.flatten()

    if output_file is not False: df.to_csv(output_file, sep='\t', index=False)
    return df

def predict_from_dataframe(input_file, models, column_sequences, L_max, output_file = False, 
                           padding='left', padding_value=0, batch_size=2000, colum_pred_name='predictions_GFP', store_variance=False,
                           adaptors=False, verbose=False, measurement_column=False, header_only=False, split_on_variable=False,
                           model_type='MTtrans',
                           # External model kwargs
                           noderer_order='di', noderer_aug_col=None,
                           noderer_positions=None,
                           label_scaler_path=None, evo2_batch_size=32,
                           evo2_regression_path=None, device='cuda', padding_with_sequence=False):
    """
    This function will predict the output of a model(s) given a dataframe with sequences.
    Args:
        input_file: (str) path to the input file
        models: (list) of models, should be the same model but different folds
        column_sequences: (str) column name of the sequences
        L_max: (int) Max length of sequences. Relevant for padding.
        output_file: (str or False) if False, it will return the dataframe with the predictions, if a string is given, it will save the dataframe to that path. (default: False)
        padding: (str) Type of padding, only 3 possible (left, middle, right). (default: middle)
        padding_value: (int or float ) Value to pad with. (default: 0)
        store_variance: (bool) If true, store variance of the models and return it

    """

    #print(f'Predicting from dataframe {input_file} using models {models} with model type {model_type}...', flush=True)

    #Create folder of output file if it doesn't exist
    if output_file is not False:
        output_folder = os.path.dirname(output_file)
        os.makedirs(output_folder, exist_ok=True)

    #Load the dataframe
    if 'xlsx' in input_file: 
        try: 
            metadata = pd.read_excel(input_file)
        except:
            metadata = pd.read_csv(input_file, sep='\t')

    else: metadata = pd.read_csv(input_file, sep='\t')
    #make sure the metadata is tab separated, otherwise do comma separated
    if metadata.shape[1] < 2: metadata = pd.read_csv(input_file, sep=';')
    if metadata.shape[1] < 2: metadata = pd.read_csv(input_file, sep=',', index_col=0)
    if metadata.shape[1] < 2: metadata = pd.read_csv(input_file, sep=';')
    
    #If there's a row in the column column_sequences that is not a string, remove it
    removed = metadata[~metadata[column_sequences].apply(lambda x: isinstance(x, str))]
    metadata = metadata[metadata[column_sequences].apply(lambda x: isinstance(x, str))]
    if not removed.empty: 
        print(f'Removed {len(removed)} rows that were not strings in the column {column_sequences}: \n {removed}', flush=True)

    #Make sure that the sequence is not longer than L_max
    metadata[f'length_{column_sequences}'] = metadata[column_sequences].apply(len)
    metadata = metadata[metadata[f'length_{column_sequences}'] <= (L_max-len(adaptors[0])-len(adaptors[1]))]

    #Get the sequences
    seqs = metadata[column_sequences].tolist()

    # ── Load external predictors ──────────────────────────────────────────────
    if model_type in ('noderer', 'evo2'):
        model_path = models[0] if isinstance(models, list) else models
        predictor  = _load_external_predictor(
            model_path, model_type, device,
            noderer_order=noderer_order, noderer_aug_col=noderer_aug_col,
            noderer_positions=noderer_positions,
            label_scaler_path=label_scaler_path,
            evo2_batch_size=evo2_batch_size,
            evo2_regression_path=evo2_regression_path,
        )

        # Gather per-row AUG indices for Noderer if the column exists
        noderer_aug_indices = None
        if model_type == 'noderer' and noderer_aug_col is not None:
            if noderer_aug_col in metadata.columns:
                noderer_aug_indices = metadata[noderer_aug_col].tolist()
            else:
                print(f"  WARNING: --noderer_aug_col '{noderer_aug_col}' not found in "
                      f"dataframe — falling back to first ATG detection.", flush=True)

        predictions = predict_from_seq(
            predictor, seqs, L_max, model_type=model_type,
            noderer_aug_indices=noderer_aug_indices,
            evo2_batch_size=evo2_batch_size, device=device,
            padding_with_sequence=padding_with_sequence
        )

    else:
        #Load each model
        loaded_models_list = []
        for model in models:
            loaded_model = load_model(model, model=model_type, train=False, verbose=verbose, L_max=L_max)
            if torch.cuda.is_available(): loaded_model = loaded_model.cuda()
            loaded_model.eval()
            loaded_models_list.append(loaded_model)
        print(f'Number of models loadded: {len(loaded_models_list)}', flush=True)
        #Predict
        if store_variance: 
            predictions, variances = predict_from_seq(loaded_models_list, seqs, L_max, padding=padding, 
                                                    padding_value=padding_value, batch_size=batch_size, 
                                                    variance_models = store_variance, adaptors=adaptors, model_type=model_type, padding_with_sequence=padding_with_sequence)
        else: 
            predictions = predict_from_seq(loaded_models_list, seqs, L_max, padding=padding, padding_value=padding_value, 
                                        batch_size=batch_size, variance_models = store_variance, adaptors=adaptors, model_type=model_type, padding_with_sequence=padding_with_sequence)

    #Print the shape
    metadata[colum_pred_name] = predictions
    if store_variance and 'variances' in dir(): 
        metadata[f'{colum_pred_name}_variance'] = variances

    if header_only: 
        #Remove Sequecne column
        metadata = metadata.drop(columns=[column_sequences])

    if output_file is not False: metadata.to_csv(output_file, sep='\t', index=False)

    #Remove the NaNs
    metadata = metadata[~metadata[colum_pred_name].isna()]


    

    
    if measurement_column:
        extension_output_file = os.path.splitext(output_file)[0]
        #if it's a measurement column, make a scatter plot of the predictions vs the measurement column
        if measurement_column in metadata.columns:
            fig, ax = plt.subplots(figsize=(7, 5))

                
            r2 = np.corrcoef(metadata[measurement_column], metadata[colum_pred_name])[0, 1]
            if False: #'Variant' in metadata.columns:
                metadata['Type'] = metadata['Variant'].fillna('').apply(lambda x: x.split('_')[-2])
                sns.scatterplot(x=metadata[measurement_column], y=metadata[colum_pred_name], ax=ax, 
                             linewidth=0, alpha=1, hue=metadata['Type'], palette='Set1', s=50)

            else:
                sns.scatterplot(x=metadata[measurement_column], y=metadata[colum_pred_name], ax=ax,
                                 linewidth=0, alpha=1, color='black', s=50)
            plt.xlabel(f'Measurement {measurement_column}')
            plt.ylabel('Predicted GFP scores')
            plt.title(f'Pearson\'s r={r2:.2f}')
            plt.legend(title='', loc='upper left', frameon=False)
            output_figure = extension_output_file + f'_scatter_{measurement_column}_vs_predictions.png'
            plt.savefig(output_figure, dpi=300, bbox_inches='tight')

            #Now make histplot
            fig, ax = plt.subplots(figsize=(7, 5))
            g = ax.hist2d(metadata[measurement_column], metadata[colum_pred_name], bins=100, cmap='afmhot', norm=colors.LogNorm())
            sns.regplot(x=metadata[measurement_column], y=metadata[colum_pred_name], ax=ax, scatter=False, color='blue')
            #Add colorbar
            cbar = plt.colorbar(g[3], ax=ax, label='Sequence counts (log scale)')
            ax.set_ylabel('Predicted GFP scores')
            plt.title(f'Pearson\'s r={r2:.2f}')
            ax.set_xlabel(f'Measurement {measurement_column}')
            output_figure = extension_output_file + '_hist2d_predictions.png'
            plt.savefig(output_figure, dpi=300, bbox_inches='tight')
        

            #Now check if the split_on_variable exists, and if so, check if it's in the metadata, and if so, make a scatter plot of the predictions vs the measurement but make column=5 and rows the remaining
            if split_on_variable and split_on_variable in metadata.columns:
                #Take only the unique values that have at least 10 values
                metadata_filtered = metadata.groupby(split_on_variable).filter(lambda x: len(x) >= 10)
                unique_values = metadata_filtered[split_on_variable].unique()
                #Order the unique values by the variance of the measurement column
                unique_values = sorted(unique_values, key=lambda x: metadata[metadata[split_on_variable] == x][measurement_column].var())
                #Reverse
                unique_values = unique_values[::-1]
                n_cols = 5 if len(unique_values) < 20 else 10
                n_rows = int(np.ceil(len(unique_values) / n_cols))

                size_fig_row = np.max([30, 5*n_rows])
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, size_fig_row), sharex=True, sharey=True)
                axes = axes.flatten()
                corr_var = {}
                for i, value in enumerate(unique_values):
                    subset = metadata[metadata[split_on_variable] == value]
                    r2_subset = np.corrcoef(subset[measurement_column], subset[colum_pred_name])[0, 1]
                    var_measurement = subset[measurement_column].var()
                    sns.scatterplot(x=subset[measurement_column], y=subset[colum_pred_name], ax=axes[i],
                                    linewidth=0, alpha=1, color='black', s=50)
                    axes[i].set_title(f'{split_on_variable}={value}, n={len(subset)}\n Measurements Var.={var_measurement:.2f}, Pearson r={r2_subset:.2f}')
                    axes[i].set_xlabel(f'Measurement {measurement_column}')
                    axes[i].set_ylabel('Predicted GFP scores')
                    corr_var[value] = {'r2': r2_subset, 'var_measurement': var_measurement}
                plt.tight_layout()
                output_figure = extension_output_file + f'_scatter_{measurement_column}_vs_predictions_split_by_{split_on_variable}.png'
                plt.savefig(output_figure, dpi=300, bbox_inches='tight')

                #Save the correlation and variance in a csv file
                corr_var_df = pd.DataFrame.from_dict(corr_var, orient='index')
                #Index to column
                corr_var_df = corr_var_df.reset_index().rename(columns={'index': split_on_variable})
                output_corr_var = extension_output_file + f'_correlation_variance_split_by_{split_on_variable}.txt'
                corr_var_df.to_csv(output_corr_var, sep='\t', index=False)

                #Make a barplot of all the correlations with the TIS in the x axis and the correlation in the y axis
                #Make a heatmap on top but that it's way smaller in the y-axis (i.e. 2)
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(max(20, len(unique_values)*0.5), 8), gridspec_kw={'height_ratios': [1, 5]})
                sns.barplot(x=split_on_variable, y='r2', data=corr_var_df, ax=ax[1], color='lightgray')
                #Rotate x ticks
                ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
                ax[1].set_ylabel(f'Pearson\'s r measurements vs predictions \n within sequences with the same {split_on_variable}')
                ax[1].set_xlabel(f'{split_on_variable}')
                #ax[1].set_title(f'Correlation between predicted GFP scores and measurements split by {split_on_variable}')
                #Put the correlation value on top of each bar
                for i, row in corr_var_df.iterrows():
                    n = metadata[metadata[split_on_variable] == row[split_on_variable]].shape[0]
                    ax[1].text(i, row['r2']*0.98, f"{row['r2']:.2f},\nn={n}", ha='center', va='bottom', fontsize=10)

                #Add on top a heatmap that indicates 1) the number of sequences and 2) the variance of the measurements for each split_on_variable value
                #Create a new axis on top of the barplot
                ax2 = ax[0]
                sns.heatmap(corr_var_df[['var_measurement']].T, ax=ax2, cmap='Reds', cbar=False, alpha=0.5, annot=corr_var_df[['var_measurement']].T, 
                                fmt='.1f', annot_kws={'fontsize': 10})
                ax2.set_xlabel('')
                #Remove x ticks
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.set_ylabel('Measurement\n variance')

                plt.tight_layout()
                

                output_figure = extension_output_file + f'_barplot_correlation_split_by_{split_on_variable}.png'
                plt.savefig(output_figure, dpi=300, bbox_inches='tight')


                #Make also histogram of the correlation values
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(corr_var_df['r2'], bins=20, ax=ax, color='lightgray', edgecolor='black')
                ax.set_xlabel(f'Pearson\'s r measurements vs predictions \n within sequences with same {split_on_variable}')
                ax.set_ylabel('Count')
                average = corr_var_df['r2'].mean()
                #Put a line at the average value
                ax.axvline(average, color='red', linestyle='--', label=f'Average Pearson\'s\nr={average:.2f}')
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))
                ax.set_title(f'Histogram of correlation values \n split by {split_on_variable}')
                plt.tight_layout()
                output_figure = extension_output_file + f'_histogram_correlation_split_by_{split_on_variable}.png'
                plt.savefig(output_figure, dpi=300, bbox_inches='tight')

                #Make a regression plot of the correlation vs the variance of the measurements
                fig, ax = plt.subplots(figsize=(7, 5))
                #Make line red and dots black
                sns.regplot(x='var_measurement', y='r2', data=corr_var_df, ax=ax, scatter_kws={'color': 'black'}, line_kws={'color': 'red'})
                ax.set_xlabel(f'Variance of measurements \n within sequences with same {split_on_variable}')
                ax.set_ylabel(f'Pearson\'s r measurements vs predictions \n within sequences with same {split_on_variable}')
                ax.set_title(f'Correlation between measurement variance and correlation with predictions \n split by {split_on_variable}')
                r2 = np.corrcoef(corr_var_df['var_measurement'], corr_var_df['r2'])[0, 1]
                ax.text(0.05, 0.95, f'Pearson r={r2:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=10, color='red')
                plt.tight_layout()
                output_figure = extension_output_file + f'_correlation_measurement_variance_split_by_{split_on_variable}.png'
                plt.savefig(output_figure, dpi=300, bbox_inches='tight')    

                #Now do the same by number of sequences
                fig, ax = plt.subplots(figsize=(7, 5))
                corr_var_df['n_sequences'] = corr_var_df[split_on_variable].apply(lambda x: metadata[metadata[split_on_variable] == x].shape[0])
                sns.regplot(x='n_sequences', y='r2', data=corr_var_df, ax=ax, scatter_kws={'color': 'black'}, line_kws={'color': 'red'})
                ax.set_xlabel(f'Number of sequences \n with same {split_on_variable}')
                ax.set_ylabel(f'Pearson\'s r measurements vs predictions \n within sequences with same {split_on_variable}')
                ax.set_title(f'Correlation between number of sequences and correlation with predictions \n split by {split_on_variable}')
                r2 = np.corrcoef(corr_var_df['n_sequences'], corr_var_df['r2'])[0, 1]
                ax.text(0.05, 0.95, f'Pearson r={r2:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=10, color='red')
                plt.tight_layout()
                output_figure = extension_output_file + f'_correlation_number_sequences_split_by_{split_on_variable}.png'
                plt.savefig(output_figure, dpi=300, bbox_inches='tight')

                #Now make AUC of variance cutoff vs. correlation
                #For each model plot how the correlation gets better at different cutoffs of variance explained by the TIS
                cutoffs = [0, 0.1, 0.5, 1, 2, 3, 5, 10]


                cutoff_data = []
                for cutoff in cutoffs:
                    subset_data = corr_var_df[corr_var_df['var_measurement'] >= cutoff]
                    print(f'Cutoff: {cutoff}, Number of data points: {len(subset_data)}')
                    
                    avg_r2 = subset_data[f'r2'].mean()
                        
                    cutoff_data.append({'cutoff': cutoff, 'r2': avg_r2})

                cutoff_df = pd.DataFrame(cutoff_data)
                print(f'Cutoff dataframe: \n{cutoff_df}', flush=True)

                plt.figure(figsize=(7, 6))

                #import from sklearn
                #from sklearn.metrics import auc as sklearn_auc

                #auc = sklearn_auc(cutoff_df['cutoff'], cutoff_df['r2'])
                auc=np.nan
                #print(f'AUC of correlation vs. measurement variance cutoff: {auc}', flush=True)

                sns.lineplot(data=cutoff_df, x='cutoff', y=f'r2', marker='o', color='black')

                #plt.title(f'Correlation vs. Measurement Variance Cutoff\nAUC={auc:.2f}')


                plt.xlabel('Cutoff on Measurement Variance\nin sequences within same TIS')
                plt.ylabel('Average Pearson\'s r \npredictions vs measurements')
                plt.grid()
                plt.savefig(extension_output_file + f'_correlation_vs_measurement_variance_cutoff_split_by_{split_on_variable}.png', dpi=300, bbox_inches='tight')

    return metadata

