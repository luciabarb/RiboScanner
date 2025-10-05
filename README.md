
# Introduction

eTIS model is a deep learning model that predicts leaky scanning given a 5'UTR sequence. The sequence should include the putative Translation Initiation Site (TIS) and the surrounding sequence.

# Installation

Optionally, create a new environment for **eTIS_model** :

```
conda create -n eTIS_model python=3.9
conda activate eTIS_model
```

Then, you can install directly from GitHub:

``` pip install git+https://github.com/luciabarb/eTIS_model.git ```

To verify that eTIS_model was installed correctly, run the following command. It should display the help message without errors:

``` eTIS_model --help``` 

# Usage examples

### Predicting leaky scanning

To predict GFP levels associated with leaky scanning for each sequence in a tab-separated dataframe, run:

```sh
eTIS_model predict \
 --input ./example_data/input.txt \
 --column_sequence sequence \
--output ./output.txt
```

To use a FASTA file instead, simply provide it as the argument to --input.

> The argument `--column_sequence` should be the column in your dataframe that includes the sequences to predict.
> Note that you should replace `./example_data/input.txt` with the actual path to the file available on this page.

The output is a tab-separated file.
The first columns are identical to those provided in the input dataframe, followed by the sequence length (`length_sequence`) and the predicted GFP levels (`predictions_GFP`).

For the command line above, you should expect the following result:

| sequence    | sequence    | length_sequence                 | prediction_K562   |
-------------|-------------|----------------------------------|-------------------|
| 1          | ATGGAAAG... | 44                               | 14.12923          |
| 2          | ATAAAATA... | 73                               | 0.20971887        |
| 3          | 	AGAAGCC... | 73                               | 5.6288967         |


## Citation

If you make use of eTIS model and/or this pipeline, please cite:
