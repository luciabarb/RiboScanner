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
