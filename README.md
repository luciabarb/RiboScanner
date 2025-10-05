
# Introduction

eTIS model is a deep learning model that predicts leaky scanning given a 5'UTR sequence. The sequence should include the putative Translation Initiation Site (TIS) and the surrounding sequence.

# Installation

# Predictions - Usage examples

To predict GFP levels associated with leaky scanning for each sequence in a tab-separated dataframe, run:

```sh
eTIS_model predict \
 --model ./pretrained_models/*.pth \ 
 --input ./example_data/input.txt \ 
 --column_sequence sequence \
--output ./output.txt
```

To use a FASTA file instead, simply provide it as the argument to --input.

> Note that you should replace `pretrained_models/*.pth` with the actual path to the pre-trained models available on this page.

The output is a tab-separated file.
The first columns are identical to those provided in the input dataframe, followed by the sequence length (`length_sequence`) and the predicted GFP levels (`predictions_GFP`).

For the command line above, you should expect the following result:

| sequence    | sequence    | length_sequence                 | prediction_K562   |
-------------|-------------|----------------------------------|-------------------|
| 1          | ATGGAAAG... | 44                               | 14.12923          |
| 2          | ATAAAATA... | 73                               | 0.20971887        |
| 3          | 	AGAAGCC... | 73                               | 5.6288967         |
