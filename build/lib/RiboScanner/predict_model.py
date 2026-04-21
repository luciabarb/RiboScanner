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


def predict_from_seq(models, seqs, L_max, padding='left', padding_value=0, batch_size=2000, 
                        variance_models = False, adaptors=False, verbose=False):
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

    if type(models) != list: models = [models]
    if type(seqs) != list: seqs = [seqs]

    #Split the sequences in batches
    batches = [seqs[i:i+batch_size] for i in range(0, len(seqs), batch_size)]

    #Loop over the batches
    predictions = []
    if variance_models: variances = []

    if verbose:
        batches_loader = tqdm(batches, total = len(batches), ncols=80)
    else: batches_loader = batches
    
    for batch in batches_loader:
        pred_models = []
        if variance_models: var_models = []
        #Transform the sequences to one hot
        onehot = getOneHot(batch, L_max, padding = padding, padding_value=padding_value, adaptors=adaptors)
        #Transform to tensor and to gpus if available
        onehot = torch.tensor(np.float32(onehot))
        onehot = onehot.permute(0,2,1)
        if torch.cuda.is_available(): onehot = onehot.cuda()
        #Now loop over the models
        for model in models:
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
                       store_variance=False, padding='left', padding_value=0, batch_size=2000, adaptors=False, verbose=False, header_only=False):
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

    #Load each model
    loaded_models_list = []
    for model in models:
        loaded_model = load_model(model, model='MTtrans', train=False, verbose=verbose)
        if torch.cuda.is_available(): loaded_model = loaded_model.cuda()
        loaded_model.eval()
        loaded_models_list.append(loaded_model)
    print(f'Number of models loadded: {len(loaded_models_list)}', flush=True)

    #Predict
    #Predict
    if store_variance: 
        predictions, variances = predict_from_seq(loaded_models_list, seqs, L_max, padding=padding, 
                                                padding_value=padding_value, batch_size=batch_size, 
                                                variance_models = store_variance, adaptors=adaptors)
    else: 
        predictions = predict_from_seq(loaded_models_list, seqs, L_max, padding=padding, padding_value=padding_value, 
                                       batch_size=batch_size, variance_models = store_variance, adaptors=adaptors)
    
    #Return a dataframe with the headers and the predictions
    if header_only:df = pd.DataFrame({'header': headers, 'predictions': predictions.flatten()})
    else: df = pd.DataFrame({'header': headers, 'sequence': seqs, 'predictions': predictions.flatten()})
    if store_variance: df['variance'] = variances.flatten()
    if output_file is not False: df.to_csv(output_file, sep='\t', index=False)
    return df

def predict_from_dataframe(input_file, models, column_sequences, L_max, output_file = False, 
                           padding='left', padding_value=0, batch_size=2000, colum_pred_name='predictions_GFP', store_variance=False,
                           adaptors=False, verbose=False, measurement_column=False, header_only=False):
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

    #Load each model
    loaded_models_list = []
    for model in models:
        loaded_model = load_model(model, model='MTtrans', train=False, verbose=verbose)
        if torch.cuda.is_available(): loaded_model = loaded_model.cuda()
        loaded_model.eval()
        loaded_models_list.append(loaded_model)
    print(f'Number of models loadded: {len(loaded_models_list)}', flush=True)
    #Predict
    if store_variance: 
        predictions, variances = predict_from_seq(loaded_models_list, seqs, L_max, padding=padding, 
                                                padding_value=padding_value, batch_size=batch_size, 
                                                variance_models = store_variance, adaptors=adaptors)
    else: 
        predictions = predict_from_seq(loaded_models_list, seqs, L_max, padding=padding, padding_value=padding_value, 
                                       batch_size=batch_size, variance_models = store_variance, adaptors=adaptors)

    #Print the shape
    metadata[colum_pred_name] = predictions
    if store_variance: metadata[f'{colum_pred_name}_variance'] = variances

    if header_only: 
        #Remove Sequecne column
        metadata = metadata.drop(columns=[column_sequences])

    if output_file is not False: metadata.to_csv(output_file, sep='\t', index=False)


    

    
    if measurement_column:
        #if it's a measurement column, make a scatter plot of the predictions vs the measurement column
        if measurement_column in metadata.columns:
            fig, ax = plt.subplots(figsize=(7, 5))

                
            r2 = np.corrcoef(metadata[measurement_column], metadata[colum_pred_name])[0, 1]
            if 'Variant' in metadata.columns:
                metadata['Type'] = metadata['Variant'].fillna('').apply(lambda x: x.split('_')[-2])
                sns.scatterplot(x=metadata[measurement_column], y=metadata[colum_pred_name], ax=ax, 
                             linewidth=0, alpha=1, hue=metadata['Type'], palette='Set1', s=50)

            else:
                sns.scatterplot(x=metadata[measurement_column], y=metadata[colum_pred_name], ax=ax,
                                 linewidth=0, alpha=1, color='black', s=50)
            plt.xlabel(f'Measurement {measurement_column}')
            plt.ylabel('Predicted GFP scores')
            plt.title(f'Pearson correlation coefficient={r2:.2f}')
            plt.legend(title='', loc='upper left', frameon=False)
            output_figure = extension_output_file + f'_scatter_{measurement_column}_vs_predictions.png'
            plt.savefig(output_figure, dpi=300, bbox_inches='tight')

            #Now make histplot
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist2d(metadata[measurement_column], metadata[colum_pred_name], bins=100, cmap='afmhot', norm=colors.LogNorm())
            ax.set_xlabel('Predicted GFP scores')
            plt.title(f'Pearson correlation coefficient={r2:.2f}')
            ax.set_ylabel(f'Measurement {measurement_column}')
            output_figure = extension_output_file + '_hist2d_predictions.png'
            plt.savefig(output_figure, dpi=300, bbox_inches='tight')

    return metadata

