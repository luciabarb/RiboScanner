##Train model 
##python 

import sys
import os
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import argparse
from scipy.stats import pearsonr
import time
import math

global today, output_folder
today = time.strftime("%Y-%m-%d").replace('-','')

from tqdm import tqdm
from matplotlib import pyplot as plt, colors
import matplotlib as mpl

from .utils_model import load_model, dataset_batch_onehot, WeightedMultiOutputMSELoss

params_figs = {'legend.fontsize': 'x-large',
         'axes.titlesize':'x-large',
         'axes.linewidth': 2,
         'axes.labelsize' : 'x-large',
         'ytick.major.width': 2,
         'ytick.minor.width': 2,

         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

mpl.rcParams.update(params_figs)
#sns.set(font_scale = 1.5)

#Define arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type = str)

    parser.add_argument('--input_train_data',  
                    help = 'Path with files to the training data, there should be the fold of the data', nargs='+')

    parser.add_argument('--model_input', type = str, default = None, help = 'Path to the model')
    
    parser.add_argument('--lr', type = float, default = 0.0005, help = 'Learning rate')
    
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size')
    
    parser.add_argument('--num_workers', type = int, default = 1, help = 'Number of workers')
    
    parser.add_argument('--type_padding', type = str, default = 'random', help = 'Type of padding, possibilities are right, left, middle, and random', 
                        choices = ['right', 'left', 'middle', 'random'])
    
    parser.add_argument('--padding_value', type=int, default = 0, help = 'Value for padding')

    parser.add_argument('--padding_with_sequence', type = bool, default = False, help = 'If True, the padding will be with a random sequence, otherwise with a value'
                                                                                ' provided in padding_value, value or random. Bool value, provide True or False')
    
    parser.add_argument('--L_max', type = int, default = 156, help = 'Max length of the sequence')
    
    parser.add_argument('--epochs', type = int, default = 25, help = 'Number of epochs')
    
    parser.add_argument('--gradient_clipping', type = float, default = False, help = 'Gradient clipping')
    
    parser.add_argument('--betas',  type=float, nargs='+', default = [0.05,0.05], help = 'Regularization terms, L1 and L2 respectively')
    
    parser.add_argument('--column_labels', type = str, nargs='+', default = ['mean_GFP_nolog2'],
                        help = 'Column label(s) of the data. Provide a single column name for regular single-head '
                                'training, or two column names (space-separated) for two-head training, e.g. '
                                '--column_labels mean_GFP_nolog2 ribosomal_load. When two columns are given, samples are '
                                'allowed to have a value for only one of the two columns (the other one can be NaN); '
                                'the loss for each head is only computed on the samples that have that measurement.')
    
    parser.add_argument('--column_sequences', type = str, default = 'Sequence', help = 'Column with the sequences')
    
    parser.add_argument('--model_architecture', type = str, default = 'MTtrans', help = 'Model architecture', choices = [ 'MTtrans', 'GemoRNA', 'dense_layers'])
    
    parser.add_argument('--criterion', type = str, default = 'mse', help = 'Criterion for the loss', choices = ['mse', 'poisson', 'SmoothL1Loss'])

    parser.add_argument('--weight_loss', type = float, default =False, nargs = '+', help = 'Weight for the loss, if there are two outputs, provide two values, one for each output')
    
    parser.add_argument('--scheduler', type=bool, default=False, help = 'Use scheduler for the learning that changes lr')
    

    parser.add_argument('--adaptors', type=str, nargs='+', default=['AGTGAACC', 'GGCGGCAG'], help='adaptors sequences, several can be given separated by space')
    

    parser.add_argument('--algorithm_interpretation', type=str, default='ISM', choices = ['ISM', 'DeepLift'], 
                        help = 'Algorithm to interpret the sequences, ISM or DeepLift')
    

    return parser.parse_args()

def training_step(train_dataloader, model, criterion, optimizer, scheduler=False, betas = (False, False), 
                    gradient_clipping=False, n_outputs=1):

    """
    Training loop.

    Args:
        train_dataloader: Train data in torch dataloader
        Model: Pytorch model
        criterion: (fun) loss function
        optimizer:
        scheduler:
        betas: (tuple) (int, int) Beta 1 and Beta 2 respectively for regularization.
        
    Returns:
        y_train_predicted: (np.array) Fragment predictions
        y_train_true: (np.array) Measured SuRE score, matching fragments with the one in y_train_predicted
        training_loss: (float) Loss performance of epoch.
    """

    model.train()

    training_loss = 0.0
    y_train_predicted, y_train_true = np.empty((0, n_outputs)), np.empty((0, n_outputs))
    
    total_number_batches = len(train_dataloader)
    #Loop through baches
    for batch_ndx, (X) in tqdm(enumerate(train_dataloader), total= len(train_dataloader), ncols=100):
        optimizer.zero_grad()

            
        X, y = X[0], X[1]

        #If X has more than 2 dimensions
        if len(X.shape) > 2:
            dims = list(range(len(X.shape)))
            dims[-1], dims[-2] = dims[-2], dims[-1]
            X = X.permute(*dims)
                
        """if n_outputs == 1:
            y = y.unsqueeze(1)"""
        

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        pred = model(X)


        y_train_predicted = np.append(y_train_predicted, pred.cpu().detach().numpy(), axis=0) #Remove the .flatten()
        y_train_true = np.append(y_train_true, y.cpu().detach().numpy(), axis=0) #Remove the .flatten()
        
        #If the prediction has two nodes, we need to mask the loss, because some samples might not have a value for one of the two heads
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            #mask = 1 where the label for that head is present, 0 where it's NaN/missing
            mask = (~torch.isnan(y)).float()
            y = torch.nan_to_num(y, nan=0.0)   

        if len(pred.shape) > 1 and pred.shape[1] > 1:
            loss = criterion(pred, y, mask)
        else:
            loss = criterion(pred, y)
            
        if betas[0] != 0 or betas[1] != 0: #If there's regularization terms, add penalty in the model

            l2_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())
            l1_norm = sum(torch.sum(torch.abs(weight)) for name, weight in model.named_parameters())
            #l1_norm = sum(torch.norm(weight, p=1) for name, weight in model.named_parameters())
            l1_norm = 0
            
            loss = loss  + l2_norm*betas[1] + l1_norm*betas[0]


        # Backpropagation

        loss.backward()

        if gradient_clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

        optimizer.step()

        training_loss += loss.item()


        #If there's NaN values, stop the training, there's something wrong
        if math.isnan(loss.item()):
            print(f' Something going wrong, loss with NaN values\n      Y: {y} \n       pred: {pred} \n       X: {X}', flush=True) 
            print(f' Mask {mask} \n Loss {loss} \n Training loss {training_loss}', flush=True)
            exit()

        if scheduler: scheduler.step() #If there's an scheduler, the learning rate need to be optimized

        
        #Print results so far, only ten batches per epoch
        batch_to_print = range(0, total_number_batches+1, total_number_batches//50)
        if batch_ndx in batch_to_print:
            loss, current = training_loss/(batch_ndx+1) , batch_ndx * len(X)
            perc = current/(len(train_dataloader)*X.shape[0])*100

            #print(f"                         loss: {loss:>7f}  [{current}/{(len(train_dataloader)*X.shape[0])}]  {round(perc,3)}%", flush=True)
            if gradient_clipping: 
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clipping))
                print(f"                         Total gradient norm: {total_norm.item():>8f}", flush=True)

            for param_group in optimizer.param_groups:
                print(f"                         Learning rate: {param_group['lr'] }", flush=True)
                continue

        
    training_loss /= ((batch_ndx))

    print(f"                              Training Error: Avg loss: {training_loss:>8f}", flush=True)

    #Print gradient
    if gradient_clipping:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clipping))
        print(f"                              Total gradient norm: {total_norm.item():>8f}", flush=True)
        #print also the max value of the 


    mse = np.nanmean(((y_train_predicted-y_train_true)**2)**(1/2))
    print(f"                                       MSE {mse:>3f} \n", flush=True)
    return(y_train_predicted, y_train_true, training_loss)

def evaluation_step(valid_dataloader, model, criterion, optimizer, type_data='Validation', n_outputs=1):
    """
    Validation loop.
    Args:
        valid_dataloader:
        model:
        criterion:
       
    Returns:
        y_val_predicted: (np.array) Fragment predictions
        y_valid_true: (np.array) Measured SuRE score, matching fragments with the one in y_train_predicted
        valid_loss: (float) Loss performance of epoch.
    """

    #Set model to evaluation mode
    #Set gradients to zero
    optimizer.zero_grad()

    y_val_predicted, y_val_real =  np.empty((0, n_outputs)), np.empty((0, n_outputs))

    model.eval()

    val_loss = 0.0


    with torch.no_grad():
        for batch_ndx, (X) in tqdm(enumerate(valid_dataloader), total= len(valid_dataloader), ncols=100):
            X, y = X[0], X[1]

            #If X has more than 2 dimensions
            if len(X.shape) > 2:
                dims = list(range(len(X.shape)))
                dims[-1], dims[-2] = dims[-2], dims[-1]
                X = X.permute(*dims)
            #X = X.permute(0,2,1)

            """if n_outputs == 1:
                y = y.unsqueeze(1)"""

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            pred = model(X)

            y_val_predicted = np.append(y_val_predicted, pred.cpu().detach().numpy(), axis=0)
            y_val_real = np.append(y_val_real, y.cpu().detach().numpy(), axis=0)

            #If the prediction has two nodes, we need to mask the loss, because some samples might not have a value for one of the two heads
            if len(pred.shape) > 1 and pred.shape[1] > 1:
                #mask = 1 where the label for that head is present, 0 where it's NaN/missing
                mask = (~torch.isnan(y)).float()
                y = torch.nan_to_num(y, nan=0.0)   

                loss = criterion(pred, y, mask)
            
            else:
                loss = criterion(pred, y)

            val_loss += loss.item()

            


    val_loss /= (batch_ndx)

    print(f"                              Testing mode {type_data} Error: Avg loss: {val_loss:>8f}", flush=True)

    mse = np.nanmean(((y_val_predicted-y_val_real)**2)**(1/2))
    print(f"                                             MSE {mse:>3f} \n", flush=True)
    
    
    return(y_val_predicted, y_val_real, val_loss)


def plot_pred_vs_true(y_true, y_val, output_folder, title, column_labels):
    """
    Plot the prediction vs the true value
    Args:
        y_true: (np.array) True values
        y_val: (np.array) Predicted values
        output_folder: (str) Output folder
        column_labels: (str) Column label of the data
        title: (str) Title of the plot
    """
    mpl.rcParams.update(params_figs)

    if len(column_labels) > 1:
        r_values, pvalues = [], []

    for n_label in range(len(column_labels)):

        y_true_label = y_true[:, n_label]
        y_val_label = y_val[:, n_label]
        #Remove the nan if there are 
        mask = ~np.isnan(y_true_label) 
        mask2 = ~np.isnan(y_val_label)
        mask = mask & mask2
        y_true_label = y_true_label[mask]
        y_val_label = y_val_label[mask]


        g= sns.jointplot(x=y_true_label,y=y_val_label, kind='hex', gridsize=40, cmap='afmhot_r', marginal_kws=dict(bins=75, fill=True, color='black'))

        g.ax_joint.hist2d(y_true_label, y_val_label, bins=(40, 40), norm=colors.LogNorm(), cmap='afmhot_r' )

        g.fig.suptitle(title.replace('_', ' '))

        values_pearsonr = pearsonr(y_val_label, y_true_label)
        #Make pvalue in scientific notation 
        pvalue = '{:.1e}'.format(values_pearsonr[1])

        rvalue = '%.2f' % (values_pearsonr[0])

        if len(column_labels) > 1:
                r_values.append(float(rvalue))
                pvalues.append(float(pvalue))

        #Take the ax of jointplot
        ax = g.ax_joint
        r = np.corrcoef(y_true_label, y_val_label)[0,1]

        r2 = np.corrcoef(y_true_label, y_val_label)[0,1]
        #round

        r2 = '%.2f' % r2


        ax.text(0.2, 0.9, f'r = {rvalue}, \n pvalue={pvalue} \n  n= {len(y_val_label)}\n r2={r2}', horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes, fontsize=10)
        sns.regplot(y=y_val_label, x=y_true_label, ax=ax, scatter=False, color='black')
        #h = ax.hist2d(y_true_label, y_val_label, bins=75, cmin=1, norm = LogNorm(), cmap = 'afmhot_r')

        #Add colorbar, far from plot

        ax.set_xlabel(f'Measured {column_labels[n_label].replace("_", " ")}')
        ax.set_ylabel(f'Prediction {column_labels[n_label].replace("_", " ")}')


        if len(column_labels) == 1:
            plt.savefig(os.path.join(output_folder, f'{today}_{title}.png'), bbox_inches='tight')
        
        else:
            plt.savefig(os.path.join(output_folder, f'{today}_{title}_{column_labels[n_label]}.png'), bbox_inches='tight')

        print(f'        - {title},r = {rvalue}, {pvalue}', flush=True)

        plt.close()
        plt.clf()
    
    if len(column_labels) > 1: rvalue, pvalue = r_values, pvalues
    else: rvalue, pvalue = float(rvalue), float(pvalue)

    return(rvalue, pvalue)

def final_bar_plot(data, output_folder, today):
    #Save the data
    data.to_csv(os.path.join(output_folder, f'{today}_correlation_data.txt'), index=False, sep='\t')
    
    if 'Output' not in data.columns:
        fig, ax = plt.subplots(figsize= (4, 5))
        sns.barplot(ax=ax, x="Set", y="Pearson correlation coefficient", data=data, errorbar=("pi", 50), capsize=.4,
                        err_kws={"color": ".5", "linewidth": 2.5},
                        linewidth=2.5, edgecolor=".5", facecolor=(0, 0, 0, 0))
                        
        sns.stripplot(ax=ax, x="Set", y="Pearson correlation coefficient", data=data, jitter=True, color="black")
        #Add in a text the mean and std of the correlation for each set
        for i, set_name in enumerate(data['Set'].unique()):
            mean = data[data['Set'] == set_name]['Pearson correlation coefficient'].mean()
            std = data[data['Set'] == set_name]['Pearson correlation coefficient'].std()
            max_y = data[data['Set'] == set_name]['Pearson correlation coefficient'].max()
            ax.text(i, max_y+0.1, f'{mean:.2f}\n±\n{std:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=10)
    else:
        fig, ax = plt.subplots(figsize= (5, 5))
        sns.barplot(ax=ax, x="Set", y="Pearson correlation coefficient", hue="Output", data=data, errorbar=("pi", 50), capsize=.4, palette="Paired")         
        sns.stripplot(ax=ax, x="Set", y="Pearson correlation coefficient", hue="Output", data=data, color="black", jitter=False, dodge=True)
        #Add in a text the mean and std of the correlation for each set if it's split by hue
        for i, set_name in enumerate(data['Set'].unique()):
            for j, output_name in enumerate(data['Output'].unique()):
                mean = data[(data['Set'] == set_name) & (data['Output'] == output_name)]['Pearson correlation coefficient'].mean()
                std = data[(data['Set'] == set_name) & (data['Output'] == output_name)]['Pearson correlation coefficient'].std()
                #Take the max position of the y axis and put the text above it, with a small offset depending on the number of outputs
                max_y = data[(data['Set'] == set_name) & (data['Output'] == output_name)]['Pearson correlation coefficient'].max()
                ax.text(i - 0.2 +(j)*0.4, max_y+0.1, f'{mean:.2f}\n±\n{std:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        #Put it outside the plot
        ax.legend(handles[0:2], labels[0:2], frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, ax.get_ylim()[1]*1.15)
    plt.savefig(os.path.join(output_folder, f'{today}_correlation_folds.pdf'), bbox_inches='tight')


def plot_loss(epoch, loss_training, loss_validation, output_folder, i_fold, today):
    fig, ax = plt.subplots()
    ax.plot(list(range(epoch)), loss_training, label = 'Training', color = 'blue')
    ax.plot(list(range(epoch)), loss_validation, label = 'Validation', color = 'red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_folder, f'{today}_{i_fold}_loss.pdf'), bbox_inches='tight')

#main function
def main_training(args):


    #Define criterion
    if args.criterion == 'mse':
        if len(args.column_labels) > 1:
            weights = args.weight_loss if args.weight_loss else [1.0] * len(args.column_labels)
            if len(weights) != len(args.column_labels):
                raise ValueError(f'Length of weight_loss {len(weights)} is not equal to the number of outputs {len(args.column_labels)}')
            criterion = WeightedMultiOutputMSELoss(weights=weights)
        else:
            criterion = torch.nn.MSELoss()
    elif args.criterion == 'poisson': criterion = torch.nn.PoissonNLLLoss(log_input=False)
    elif args.criterion == 'SmoothL1Loss': criterion = torch.nn.SmoothL1Loss()
    
    

    output_folder = args.output_folder

    #Make all possible combinations of the file where one is the validation and the rest the training
    files_training = args.input_data

    corr_fold_train, corr_fold_valid = [], []

    for i_fold, validation_file in enumerate(files_training):
        print(f'############################################')
        print(f'\n    - Fold {i_fold}\n', flush=True)

        ###########Load model
        model = load_model(args.model_input, model=args.model_architecture, train=True, L_max=args.L_max, output_size = len(args.column_labels))
    

        if torch.cuda.is_available(): model = model.cuda()

        params_model = model.parameters()
        
        #print model in file
        with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
            f.write(f'\n        --------------------------------------------\n')
            f.write(f'        - Model: \n\t\t\t {model}\n')

            #Print total number of parameters
            #f.write(f'        - Total number of parameters: {total_params}\n')
            f.write(f'        --------------------------------------------\n')
        
        #Define optimizer
        optimizer = torch.optim.SGD(params_model, lr = args.lr,  momentum=0.9,  weight_decay = 1e-4)
        
        ###########Load data

        #Training files are all the files except the validation file
        train_files = [x for x in files_training if x != validation_file]
        print(f'        Fold {i_fold} \n\t\t- Training files: {train_files}\n\n\t\t - Validation file: {validation_file}', flush=True)

        ###########Load sampler and data
        

        #Load data
        train_data = pd.read_csv(train_files[0])
        for file in train_files[1:]:
            train_data = pd.concat([train_data, pd.read_csv(file)], axis=0)

        train_data.index = range(len(train_data))
        val_data = pd.read_csv(validation_file)
        val_data.index = range(len(val_data))


        training_set = dataset_batch_onehot(train_data, args.column_labels, args.column_sequences, L_max = args.L_max, padding = args.type_padding, 
                                                    padding_value=args.padding_value, padding_with_sequence = args.padding_with_sequence, 
                                                     adaptors=args.adaptors, type_model = args.model_architecture)
                                                     
        validation_set = dataset_batch_onehot(val_data, args.column_labels, args.column_sequences, L_max = args.L_max, padding = args.type_padding,
                                                    padding_value=args.padding_value, padding_with_sequence = args.padding_with_sequence, 
                                                     adaptors=args.adaptors, type_model = args.model_architecture)

        
        ##Print size of data
        print(f'                - Training data: {len(training_set)}', flush=True)
        print(f'                - Validation data: {len(validation_set)}', flush=True)



        train_sampler = torch.utils.data.BatchSampler(range(len(training_set)), batch_size= args.batch_size, drop_last=False)
        validation_sampler = torch.utils.data.BatchSampler(range(len(validation_set)), batch_size= args.batch_size, drop_last=False)
        train_sampler_for_validation = torch.utils.data.BatchSampler(range(len(training_set)), batch_size= args.batch_size, drop_last=False)



        params_dataloader = {'num_workers': args.num_workers, 'pin_memory':False, 'shuffle':True, 'batch_size':args.batch_size}


        training_generator = torch.utils.data.DataLoader(training_set, **params_dataloader)
        validation_generator = torch.utils.data.DataLoader(validation_set,  **params_dataloader)
        training_generator_when_validating = torch.utils.data.DataLoader(training_set, **params_dataloader)

        
        if torch.cuda.is_available():
            model.cuda()
            criterion.cuda()
        
        """if args.scheduler:
            steps_per_epoch = len(training_generator)
            total_steps = args.epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=args.lr * 0.01, last_epoch=-1
            )
            print(f'NEW SCHEDUELER WITH FIX ED ETA_MIN {args.lr*0.01} and {total_steps} steps', flush=True)
        else:
            scheduler = False"""

        
        #Loop for epochs
        loss_epoch_train, loss_epoch_val = [], []
        for epoch in range(args.epochs):
            print(f'\n        - Epoch {epoch}', flush=True)

            #Define scheduler
            if args.scheduler:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*(len(training_set)/args.batch_size), eta_min= args.lr*0.1, last_epoch=-1)


            else: scheduler = False
        

            ##From start of the model test data
            if epoch == 0: 
                y_train_predicted, y_train_true, val_loss = evaluation_step(training_generator, model, criterion, optimizer, n_outputs=len(args.column_labels))
                plot_pred_vs_true(y_train_true, y_train_predicted, output_folder, 
                                                                title=f'Training_fold{i_fold}_epoch{epoch}_without_training', column_labels= args.column_labels)
                

            #Train model
            y_train_predicted, y_train_true, training_loss = training_step(training_generator, model, criterion, optimizer, scheduler=scheduler, betas = args.betas, 
                                                                            gradient_clipping=args.gradient_clipping, n_outputs=len(args.column_labels))

            #Evaluate model
            y_val_predicted, y_val_real, val_loss = evaluation_step(validation_generator, model, 
                                                                                                criterion, optimizer, type_data='Validation', n_outputs=len(args.column_labels))

            loss_epoch_train.append(training_loss)
            loss_epoch_val.append(val_loss)
            
            #Save model and plot predictionss
            if args.epochs >= 10: epochs_to_motif = range(0, args.epochs+1, args.epochs//10)
            else: epochs_to_motif = range(0, args.epochs+1, 1)

            if epoch in epochs_to_motif or epoch == (args.epochs-1):
                torch.save(model.state_dict(), os.path.join(output_folder, f'LB{today}_model_fold{i_fold}_epoch_{epoch}.pth'))
                rvalue_train, pvalue = plot_pred_vs_true(y_train_true, y_train_predicted, output_folder, title=f'Training_fold{i_fold}_epoch{epoch}', column_labels= args.column_labels)
                rvalue_val, pvalue = plot_pred_vs_true(y_val_real, y_val_predicted, output_folder, title=f'Validation_fold{i_fold}_epoch{epoch}', column_labels= args.column_labels)


        #Plot of loss
        print(f'loss_epoch_train {loss_epoch_train} \nloss_epoch_val {loss_epoch_val}', flush=True)
        plot_loss(args.epochs, loss_epoch_train, loss_epoch_val, output_folder, i_fold, today)

        #Save loss
        corr_fold_train.append((rvalue_train))
        corr_fold_valid.append((rvalue_val))
        
    
    #Make barplots of the correlation
    mpl.rcParams.update(params_figs)

    if len(args.column_labels) == 1:
        data = {'Pearson correlation coefficient': corr_fold_train + corr_fold_valid, 'Set': ['Train']*len(corr_fold_train) + ['Validation']*len(corr_fold_valid)}
    
    else:
        n_outputs = len(args.column_labels)
        pcc = []
        corr_fold_train = np.asarray(corr_fold_train)
        corr_fold_valid = np.asarray(corr_fold_valid)
        for n_label in range(n_outputs):
            pcc.extend([corr_fold_train[:,n_label], corr_fold_valid[:,n_label]])
            #print(f'[corr_fold_train[:,n_label], corr_fold_valid[:,n_label]] {[corr_fold_train[:,n_label], corr_fold_valid[:,n_label]]}', flush=True)
        #Make it flat
        pcc = [item for sublist in pcc for item in sublist]


        set = [['Train']*len(corr_fold_train) + ['Validation']*len(corr_fold_valid)]
        #Make it flat
        set = [item for sublist in set for item in sublist]


        output = [ [name] * (len(corr_fold_train) + len(corr_fold_valid)) for name in args.column_labels ]
        #Make it flat
        output = [item for sublist in output for item in sublist]
        data = {'Pearson correlation coefficient': pcc, 'Set': set*n_outputs,
                'Output': output}
        
    
    final_bar_plot(pd.DataFrame(data), output_folder, today)



def call_main(args):
    
    if 'N' in args.adaptors[0] or args.adaptors[0] == 'false' or args.adaptors[0] == 'False' or args.adaptors[0] == 'none' or args.adaptors[0] == 'None':
        args.adaptors = False
        print(f'    - No adaptors will be used', flush=True)

    #Save data
    output_folder = os.path.join(args.output_folder, args.model_architecture)
    for trial in range(1000):
        output_folder_current = os.path.join(output_folder, f'trial_{trial}')
        if not os.path.exists(output_folder_current): 
            output_folder = output_folder_current
            break
    
    #Create folder and parents
    os.makedirs(output_folder, exist_ok=True)

    print(f'    - Output folder: {output_folder}', flush=True)
    
    with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
        f.write('Human sequences only')
        for arg in vars(args):
            f.write(f'        - {arg}: {getattr(args, arg)}\n')



    #All printing messages will be saved in a log file
    sys.stdout = open(os.path.join(output_folder, 'log_messages.txt'), 'w')
    print(f'    - Output folder: {output_folder}', flush=True)
    print(f'    - Log file: {os.path.join(output_folder, "log.txt")}', flush=True)

    args.output_folder = output_folder

    main_training(args)

    sys.stdout.close()

if __name__ == '__main__':

    args = parse_args()
    call_main(args)
