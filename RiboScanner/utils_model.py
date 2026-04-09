#Main functions to run deep learning model
import torch 
import torch.nn as nn
import numpy as np 
import random


###DATA PROCESSING
def getOneHot(Seqs, L_max, padding = 'random', padding_value=0, padding_with_sequence=False, return_reverse_complement=False, 
                    add_kozak=False, adaptors=False, position_kozak=13):
    """
    Transform list of sequences to one hot.
    Args:
        Seqs: (list) of string sequences
        L_max: (int) Max length of sequences. Relevant for padding.
        padding: (str) Type of padding, only 3 possible (left, middle, right). (default: middle)
        padding_value: (int or float ) Value to pad with. (default: 0)
        padding_type: (Bool) If True, we are going to pad with a random sequence. (default: False)
        return_reverse_complement: (Bool) If True, we are going to return the reverse complement of the sequence. (default: False)

    Returns:
        X_OneHot: (np.array) Array with length (samples, L_max, 4)
    """
    # Define nucleotide to vector
    letter2vector = {'A':np.array([1.,0.,0.,0.]),'C':np.array([0.,1.,0.,0.]),
                     'G':np.array([0.,0.,1.,0.]),'T':np.array([0.,0.,0.,1.]),
                     'N':np.array([0.,0.,0.,0.])};


    # get On Hot Encoding
    X_OneHot = []
    for seq in Seqs:
        #print(f'Processing sequence: {seq}', flush=True)
        if add_kozak:
            #If add_kozak is a list, there are several sequences to add
            if isinstance(add_kozak, list):
                if isinstance(position_kozak, list): #If position_kozak is a list, we have to add the sequences in different positions
                    if len(position_kozak) != len(add_kozak): raise ValueError('Did you give the subsequence and position correctly? Subsequence and position must have the same length')
                    else:
                        added_pos = 0
                        for subseq, pos in zip(add_kozak, position_kozak):
                            seq = seq[:(pos+added_pos)] + subseq + seq[(pos+added_pos):]
                            #added_pos += len(subseq)
            
            elif isinstance(add_kozak, str):
                seq = seq[:-position_kozak] + add_kozak + seq[-position_kozak:]
            
        if adaptors:
            seq = adaptors[0] + seq + adaptors[1]
        
        
            
        diff = (L_max - len(seq))
        pw = (L_max - len(seq))/2

        #If padding value is not 0, and it's a "random" then we will create a random sequence
        if padding_with_sequence:
            if padding == 'random':
                left_random = random.randint(0,diff)
                right_random = diff - left_random
                seq_left = ''.join(random.choices(['A','C','G','T'], k=left_random))
                seq_right = ''.join(random.choices(['A','C','G','T'], k=right_random))
                seq = seq_left + seq + seq_right
                diff, pw = 0, 0

            elif padding == 'middle':
                seq_left = ''.join(random.choices(['A','C','G','T'], k=int(np.ceil(pw))))
                seq_right = ''.join(random.choices(['A','C','G','T'], k=int(np.floor(pw))))
                seq = seq_left + seq + seq_right
                diff, pw = 0, 0
            
            elif padding == 'left':
                seq_right = ''.join(random.choices(['A','C','G','T'], k=diff))
                seq = seq + seq_right
                diff, pw = 0, 0
            
            elif padding == 'right':
                seq_left = ''.join(random.choices(['A','C','G','T'], k=diff))
                seq = seq_left + seq
                diff, pw = 0, 0

        PW = [int(np.ceil(pw)),int(np.floor(pw))]

        x = np.array([letter2vector[s] for s in seq])

        
        if padding == 'random': #If random, we padd randomly from left and right
            left_random = random.randint(0,diff)
            right_random = diff - left_random
            X_OneHot.append( np.pad(x,[[left_random,right_random],[0,0]], constant_values=padding_value) )
        
        elif padding == 'middle':
            
            X_OneHot.append( np.pad(x,[PW,[0,0]], constant_values=padding_value) )

        elif padding == 'right':
            X_OneHot.append( np.pad(x,[[0,diff],[0,0]], constant_values=padding_value) )

        elif padding == 'left':
            X_OneHot.append( np.pad(x, [[diff,0],[0,0]], constant_values=padding_value) )

        else:
            raise Exception(f'Padding option not recognised: {padding}')

    X_OneHot = np.array(X_OneHot)

    if return_reverse_complement:
        X_OneHot_RC = np.flip(X_OneHot, axis=(1,2))

        X_OneHot = np.stack([X_OneHot, X_OneHot_RC], axis=0)


    return(X_OneHot)



#############################
#DATALOADER
#############################


class dataset_batch_onehot(torch.utils.data.Dataset):
    """Dataset of sequential data to train memory.
    Args:
        pd_dataframe (pd.DataFrame): Dataframe with the data.
        column_labels (str): Column name of the dataset to use as labels.
        column_sequences (str): Column name of the dataset to use as sequences.
        L_max (int): Max length of the sequences.
        padding (str): Type of padding, only 3 possible (left, middle, right). (default: middle)
        padding_value (int or float): Value to pad with. (default: 0)
        padding_with_sequence (Bool): If True, we are going to pad with a random sequence instead of padding_value. (default: False)
        reverse_complement_seq (Bool): If True, we are going to return the reverse complement of the sequence. (default: False)
    
    Note:
        Arrays should have the same size of the first dimension and their type should be the
        same as desired Tensor type.
    """
    def __init__(self, pd_dataframe, column_labels, column_sequences, L_max, padding = 'left', padding_value=0, 
                    padding_with_sequence=False, reverse_complement_seq=False, adaptors=False):
        self.pd_dataframe = pd_dataframe
        self.column_labels = column_labels
        self.column_sequences = column_sequences
        self.L_max = L_max
        self.padding = padding
        self.padding_value = padding_value
        self.padding_with_sequence = padding_with_sequence
        self.reverse_complement_seq = reverse_complement_seq
        self.adaptors = adaptors



    def __getitem__(self, index):

        
        #Index comes in a batch as batchsampler is used

        index = np.array(index)
        
        dataframe_batch = self.pd_dataframe.iloc[index]

        #print(f'  dataframe_batch {dataframe_batch}\n\n')

        seqs = dataframe_batch[[self.column_sequences]]
        #print(f' seqs {seqs}', flush=True)

        #If seqs is not a list make it a list
        if type(seqs) == str: seqs = [seqs]
        

        #Now one hot encode
        onehot = getOneHot(seqs, self.L_max, padding = self.padding, 
                            padding_value=self.padding_value, padding_with_sequence=self.padding_with_sequence,
                            return_reverse_complement=self.reverse_complement_seq, adaptors=self.adaptors)
        onehot = torch.tensor(np.float32(onehot))

        
        labels = np.array(dataframe_batch[self.column_labels], dtype=np.float32)
        #labels = np.array(np.exp2(labels)/1000, dtype=np.float32)
        #print(f'   labels  {labels}\n\n', flush=True)
        if isinstance(labels, int): labels = np.array([labels])
        #labels = torch.Tensor(labels)

        #Print shape
        if onehot.shape[0] == 1: onehot = onehot[0]
        #print(f'onehot {onehot.shape} labels {labels}\n\n', flush=True)

        
        
        
        return onehot, labels

    def __len__(self):
        return len(self.pd_dataframe)

####################################
#MODEL FUNCTIONS
####################################

def load_model(  pretrained_model_file=None,
                    train = True, strict= False, verbose=False, model=False):
    """
    Load model depending on the name in the output_directory.

    Args:
        output_directory: (str) Directory where we want to save all output files.
        pretrained_model_file:  (str) File with model_CNN.pth file to load weights.
                                        If we don't want to pretrain model just use None.

        L_max: (int) max length of sequences
        train: (Boolean) If we are training the model or not.
        strict: (Boolean) Only relevant if there's pretrained_directory. If weights are paseted in astrict way (it has to be the same exact model) or not.
        verbose: (Boolean) If we want to print information or not.

    Return:
        model: pytorch model with initlaizied weights
    """

    model = MTtrans()
              

    #Get class name of the model
    model_name = model.__class__.__name__

    if train:

        if verbose: print(f'      Model {model_name} loaded, pretrained {pretrained_model_file}', flush=True)
        if pretrained_model_file and 'leaky_scanning_LM_UTR' not in model_name: #Load weights if there's pretrained model

            if verbose: print(f'      Model weights loaded {pretrained_model_file}', flush=True)
            
            model_weights = torch.load(pretrained_model_file, map_location=torch.device('cpu'))
            
            missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict = strict)
            

            if verbose: print(f'      Missing keys {missing_keys}, \n      Unexpected keys {unexpected_keys}', flush=True)


    else: #Validation, pretrained_model_file is mandatory
        if not pretrained_model_file: raise ValueError(f'   Argument: [pretrained_model_file] is mandatory for validation')

        model_weights = torch.load(pretrained_model_file, map_location=torch.device('cpu'))
        
        if verbose: print(f'      Model weights loaded', flush=True, end=' ')
        missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=True)
        if verbose: print('      Model weights pasted', flush=True)

        model.eval()
    
    if torch.cuda.is_available():
            if verbose: print('      Model moved to GPU', flush=True)
            model = model.cuda()

    return(model)



class Conv1d_block(nn.Module):
        def __init__(self, channel_ls, kernel_size, stride, padding_ls=None, diliation_ls=None, activation='ReLU'):
            super().__init__()

            if padding_ls is None:
                padding_ls = [0] * (len(channel_ls) - 1)
            if diliation_ls is None:
                diliation_ls = [1] * (len(channel_ls) - 1)

            self.encoder = nn.ModuleList([
                self.Conv_block(
                    channel_ls[i], channel_ls[i + 1], kernel_size, stride[i], padding_ls[i], diliation_ls[i], activation
                ) for i in range(len(channel_ls) - 1)
            ])

        def Conv_block(self, in_chan, out_chan, kernel_size, stride, padding, dilation, activation):
            activation_layer = eval(f"nn.{activation}")
            seq_conv_block = nn.Sequential(
                nn.Conv1d(in_chan, out_chan, kernel_size, stride, padding, dilation),
                nn.BatchNorm1d(out_chan),
                nn.Mish()
                #activation_layer
            )

            return seq_conv_block

        def forward(self, x):
            for block in self.encoder:
                x = block(x)
            return x
        
class MTtrans(nn.Module):
    def __init__(self):
        super().__init__()

        # Define conv_tower block directly within RL_hard_share
        self.conv_tower = Conv1d_block(
            channel_ls=[4, 128, 256, 256, 256],
            kernel_size=3,
            stride=[1, 1, 1, 3],
            padding_ls=[1, 1, 1, 1],
            diliation_ls=None,
            activation='Mish'
        )

        # Define task-specific towers with corrected naming
        self.tower = torch.nn.ModuleList([
                torch.nn.GRU(input_size=256, hidden_size=80, num_layers=2, batch_first=True),
                torch.nn.Linear(80, 1)
            ])
    

    def forward(self, x):
        # Pass through shared convolutional block
        z = self.conv_tower(x)

        # Prepare input for task-specific tower
        z_t = torch.transpose(z, 1, 2)

        # Process with GRU
        h_prim, _ = self.tower[0](z_t)

        # Final task-specific output
        out = self.tower[1](h_prim[:, -1, :])
        return out
