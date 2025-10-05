# eTIS_model

# Introduction

# Installation

# Usage examples

To predict GFP levels associated with leaky scanning for each sequence in a tab-separated dataframe, run:

```sh
eTIS_model predict \
 --model ./pretrained_models/*.pth \ 
 --input ./example_data/input.txt \ 
 --column_sequence sequence \
--output ./example_data/output.txt
```

To use a FASTA file instead, simply provide it as the argument to --input.

> Note that you should replace `pretrained_models/*.pth` with the actual path to the pre-trained models available on this page.

The output is a tab-separated file.
The first columns should be identical as the ones provided in the dataframe, followed by the length of the sequence `length_sequence` and the predictions in `predictions_GFP`.

For the command line above, you should expect the following result:
id	    	length_sequence	predictions_GFP
1	ATGGAAAGTAAATGGTAGCTCGGAAGGGTCAAAAGAGTCCGCGG	44	14.12923
2	ATAAAATAATTTTATTTTATTCAGCTTATAATATGACTCGATGGAGGAAAATTTGATAAGCATGAGAGAAGAC	73	0.20971887
3	AGAAGCCAGGGACCGGCGGTTCTGGGAGCAGCTGTGCTGGATGCCCTGGAGGAACAAGGAGGCCTCCAGTCCC	73	5.6288967

| sequence    | sequence    | length_sequence                 | prediction_K562   |
-------------|-------------|----------------------------------|-------------------|
| 1          | ATGGAAAG... | 44                               | 14.12923          |
| 2          | ATAAAATA... | 73                               | 0.20971887        |
| 3          | 	AGAAGCC... | 73                               | 5.6288967         |
