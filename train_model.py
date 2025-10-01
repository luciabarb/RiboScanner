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

from .utils_model import load_model, dataset_batch_onehot

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
    
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate')
    
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size')
    
    parser.add_argument('--num_workers', type = int, default = 1, help = 'Number of workers')
    
    parser.add_argument('--type_padding', type = str, default = 'random', help = 'Type of padding, possibilities are right, left, middle, and random', 
                        choices = ['right', 'left', 'middle', 'random'])
    
    parser.add_argument('--padding_value', type=int, default = 0, help = 'Value for padding')

    parser.add_argument('--padding_with_sequence', type = bool, default = False, help = 'If True, the padding will be with a random sequence, otherwise with a value'
                                                                                ' provided in padding_value, value or random. Bool value, provide True or False')
    
    parser.add_argument('--L_max', type = int, default = 100, help = 'Max length of the sequence')
    
    parser.add_argument('--epochs', type = int, default = 20, help = 'Number of epochs')
    
    parser.add_argument('--gradient_clipping', type = float, default = False, help = 'Gradient clipping')
    
    parser.add_argument('--betas',  type=float, nargs='+', default = [0,0], help = 'Regularization terms, L1 and L2 respectively')
    
    parser.add_argument('--column_labels', type = str, default = 'mean_GFP', help = 'Column label of the data')
    
    parser.add_argument('--column_sequences', type = str, default = 'Sequence', help = 'Column with the sequences')
    
    parser.add_argument('--model_architecture', type = str, default = 'leaky_scanning', help = 'Model architecture', choices = ['optimus_5_prime', 'leaky_scanning', 'leaky_scanning_nucleotide_transformer', 'leaky_scanning_UTR_LM', 'MTtrans'])
    
    parser.add_argument('--criterion', type = str, default = 'mse', help = 'Criterion for the loss', choices = ['mse', 'poisson'])
    
    parser.add_argument('--scheduler', type=bool, default=False, help = 'Use scheduler for the learning that changes lr')
    

    parser.add_argument('--adaptors',
                        type=str,
                        nargs='+',
                        default=['AGTGAACC', 'GGCGGCAG'],
                        help='adaptors sequences, several can be given separated by space')
    

    parser.add_argument('--algorithm_interpretation', type=str, default='ISM', choices = ['ISM', 'DeepLift'], 
                        help = 'Algorithm to interpret the sequences, ISM or DeepLift')
    

    return parser.parse_args()

def training_step(train_dataloader, model, criterion, optimizer, scheduler=False, betas = (False, False), 
                    gradient_clipping=False):

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
    y_train_predicted, y_train_true = np.array([]),np.array([])
    
    total_number_batches = len(train_dataloader)
    #Loop through baches
    for batch_ndx, (X) in tqdm(enumerate(train_dataloader), total= len(train_dataloader), ncols=100):
        optimizer.zero_grad()
            
        X, y = X[0], X[1]
        dims = list(range(len(X.shape)))
        dims[-1], dims[-2] = dims[-2], dims[-1]
        X = X.permute(*dims)
        y = y.unsqueeze(1)
        

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        pred = model(X)
        #print(  f'    pred {pred.shape} y {y.shape}')

        if betas[0] != 0 or betas[1] != 0: #If there's regularization terms, add penalty in the model

            l2_norm = sum(torch.norm(weight, p=2) for name, weight in model.named_parameters())
            l1_norm = sum(torch.sum(torch.abs(weight)) for name, weight in model.named_parameters())
            #l1_norm = sum(torch.norm(weight, p=1) for name, weight in model.named_parameters())
            l1_norm = 0
            loss = criterion(pred, y)  + l2_norm*betas[1] + l1_norm*betas[0]

        else:
            loss = criterion(pred, y)


        # Backpropagation

        loss.backward()

        if gradient_clipping: nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

        optimizer.step()

        training_loss += loss.item()

        #Check where is the maximum value, if either in one or second dimension

        y_train_predicted = np.append(y_train_predicted, pred.cpu().detach().numpy().flatten(), axis=0)
        y_train_true = np.append(y_train_true, y.cpu().detach().numpy().flatten(), axis=0)

        #If there's NaN values, stop the training, there's something wrong
        if math.isnan(loss.item()):
            print(f' Something going wrong, NaN values\n      Y: {y} \n       pred: {pred} \n       X: {X}', flush=True) 
            exit()

        if scheduler: scheduler.step() #If there's an scheduler, the learning rate need to be optimized

        
            


        #Print results so far, only ten batches per epoch
        batch_to_print = range(0, total_number_batches+1, total_number_batches//50)
        if batch_ndx in batch_to_print:
            loss, current = training_loss/(batch_ndx+1) , batch_ndx * len(X)
            perc = current/(len(train_dataloader)*X.shape[0])*100
            #print(f"                         loss: {loss:>7f}  [{current}/{(len(train_dataloader)*X.shape[0])}]  {round(perc,3)}%", flush=True)

            for param_group in optimizer.param_groups:
                #print(f"                         Learning rate: {param_group['lr'] }", flush=True)
                continue

        
    training_loss /= ((batch_ndx))

    print(f"                              Training Error: Avg loss: {training_loss:>8f}", flush=True)

    mse = np.nanmean(((y_train_predicted-y_train_true)**2)**(1/2))
    print(f"                                       MSE {mse:>3f} \n", flush=True)
    return(y_train_predicted, y_train_true, training_loss)

def evaluation_step(valid_dataloader, model, criterion, optimizer, type_data='Validation'):
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

    y_val_predicted, y_val_real =  np.array([]),np.array([])

    model.eval()

    val_loss = 0.0


    with torch.no_grad():
        for batch_ndx, (X) in tqdm(enumerate(valid_dataloader), total= len(valid_dataloader), ncols=100):
            
            X, y = X[0], X[1]
            dims = list(range(len(X.shape)))
            dims[-1], dims[-2] = dims[-2], dims[-1]
            X = X.permute(*dims)
            #X = X.permute(0,2,1)
            y = y.unsqueeze(1)

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            pred = model(X)

            loss = criterion(pred, y)

            val_loss += loss.item()

            y_val_predicted = np.append(y_val_predicted, pred.cpu().detach().numpy().flatten(), axis=0)
            y_val_real = np.append(y_val_real, y.cpu().detach().numpy().flatten(), axis=0)
            


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


    g= sns.jointplot(x=y_true,y=y_val, kind='hex', gridsize=40, cmap='afmhot_r', marginal_kws=dict(bins=75, fill=True, color='black'))

    g.ax_joint.hist2d(y_true, y_val, bins=(40, 40), norm=colors.LogNorm(), cmap='afmhot_r' )

    g.fig.suptitle(title.replace('_', ' '))

    values_pearsonr = pearsonr(y_val, y_true)
    #Make pvalue in scientific notation 
    pvalue = '{:.1e}'.format(values_pearsonr[1])

    rvalue = '%.2f' % (values_pearsonr[0])

    #Take the ax of jointplot
    ax = g.ax_joint
    r = np.corrcoef(y_true, y_val)[0,1]

    r2 = np.corrcoef(y_true, y_val)[0,1]
    #round

    r2 = '%.2f' % r2


    ax.text(0.2, 0.9, f'r = {rvalue}, \n pvalue={pvalue} \n  n= {len(y_val)}\n r2={r2}', horizontalalignment='center', 
                verticalalignment='center', transform=ax.transAxes, fontsize=10)
    sns.regplot(y=y_val, x=y_true, ax=ax, scatter=False, color='black')
    #h = ax.hist2d(y_true, y_val, bins=75, cmin=1, norm = LogNorm(), cmap = 'afmhot_r')

    #Add colorbar, far from plot

    ax.set_xlabel(f' Measured {column_labels}')
    ax.set_ylabel(f'Prediction: {column_labels}')


    plt.savefig(os.path.join(output_folder, f'LB{today}_{title}.png'), bbox_inches='tight')

    print(f'        - {title},r = {rvalue}, {pvalue}', flush=True)

    plt.close()
    plt.clf()

    return(rvalue, pvalue)

def final_bar_plot(data, output_folder, today):
    fig, ax = plt.subplots(figsize= (4, 5))
    sns.barplot(ax=ax, x="Set", y="Pearson correlation coefficient", data=data, errorbar=("pi", 50), capsize=.4,
                    err_kws={"color": ".5", "linewidth": 2.5},
                    linewidth=2.5, edgecolor=".5", facecolor=(0, 0, 0, 0))
                    
    sns.stripplot(ax=ax, x="Set", y="Pearson correlation coefficient", data=data, jitter=True, color="black")
    
    plt.ylim(0, ax.get_ylim()[1]*1.1)
    plt.savefig(os.path.join(output_folder, f'LB{today}_correlation_folds.png'), bbox_inches='tight')

def plot_loss(epoch, loss_training, loss_validation, output_folder, i_fold, today):
    fig, ax = plt.subplots()
    ax.plot(range(epoch), loss_training, label = 'Training')
    ax.plot(range(epoch), loss_validation, label = 'Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_folder, f'LB{today}_{i_fold}_loss.png'))

#main function
def main_training(args):


    #Define criterion
    if args.criterion == 'mse': criterion = torch.nn.MSELoss()
    elif args.criterion == 'poisson': criterion = torch.nn.PoissonNLLLoss(log_input=False)

        

    #Make all possible combinations of the file where one is the validation and the rest the training
    files_training = args.input_train_data

    corr_fold_train, corr_fold_valid = [], []

    for i_fold, validation_file in enumerate(files_training):
        print(f'############################################')
        print(f'\n    - Fold {i_fold}\n', flush=True)

        ###########Load model
        if 'MTtrans' in args.model_architecture:

            model = load_model(args.model_input, model=args.model_architecture, train=True,
                                freeze_known_weights=args.freeze_known_weights)
            
            #if args.model_input: model = torch.load(args.model_input, 'cpu')['state_dict']

        else: 
            model = load_model(args.model_input, train=True, L_max=args.L_max, model = args.model_architecture, poisson = args.criterion,
                            task = args.task, reverse_complement_seq=args.reverse_complement_seq)


    

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
                                                    reverse_complement_seq=args.reverse_complement_seq, adaptors=args.adaptors)
        validation_set = dataset_batch_onehot(val_data, args.column_labels, args.column_sequences, L_max = args.L_max, padding = args.type_padding,
                                                    padding_value=args.padding_value, padding_with_sequence = args.padding_with_sequence, 
                                                    reverse_complement_seq=args.reverse_complement_seq, adaptors=args.adaptors)

        
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
                y_train_predicted, y_train_true, val_loss = evaluation_step(training_generator, model, criterion, optimizer, task = args.task)
                plot_pred_vs_true(y_train_true, y_train_predicted, output_folder, 
                                                                title=f'Training_fold{i_fold}_epoch{epoch}', column_labels= args.column_labels)
                

            #Train model
            y_train_predicted, y_train_true, training_loss = training_step(training_generator, model, criterion, optimizer, scheduler=scheduler, betas = args.betas, 
                                                                            gradient_clipping=args.gradient_clipping, task = args.task)

            #Evaluate model
            y_val_predicted, y_val_real, val_loss = evaluation_step(validation_generator, model, criterion, optimizer, task = args.task, type_data='Validation')

            loss_epoch_train.append(training_loss)
            loss_epoch_val.append(val_loss)
            
            #Save model and plot predictionss
            epochs_to_motif = range(0, args.epochs+1, args.epochs//10)
            if epoch in epochs_to_motif or epoch == (args.epochs-1):
                torch.save(model.state_dict(), os.path.join(output_folder, f'LB{today}_model_fold{i_fold}_epoch_{epoch}.pth'))
                rvalue_train, pvalue = plot_pred_vs_true(y_train_true, y_train_predicted, output_folder, title=f'Training_fold{i_fold}_epoch{epoch}', column_labels= args.column_labels)
                rvalue_val, pvalue = plot_pred_vs_true(y_val_real, y_val_predicted, output_folder, title=f'Validation_fold{i_fold}_epoch{epoch}', column_labels= args.column_labels)


        #Plot of loss
        plot_loss(epoch, loss_epoch_train, loss_epoch_val, output_folder, i_fold, today)

        #Save loss
        corr_fold_train.append(float(rvalue_train))
        corr_fold_valid.append(float(rvalue_val))
        
    
    #Make barplots of the correlation
    mpl.rcParams.update(params_figs)

    data = {'Pearson correlation coefficient': corr_fold_train + corr_fold_valid, 'Set': ['Train']*len(corr_fold_train) + ['Validation']*len(corr_fold_valid)}
    final_bar_plot(pd.DataFrame(data), output_folder, today)

    

def call_main(args):
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

    main_training(args)

    sys.stdout.close()

    

if __name__ == '__main__':



    args = parse_args()

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

    main_training(args)

    sys.stdout.close()