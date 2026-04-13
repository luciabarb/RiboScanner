
# Introduction

RiboScanner is a deep learning model that given a 5'UTR sequence, predicts leaky scanning determined by the RiboScan reporter. RiboScan reporter is a highly sensitive reporter system that quantitatively measures start codon recognition with substantially greater dynamic range than conventional fluorescence reporters. 



<img width="1573" height="897" alt="draft_figure_github_RiboScanner_v2" src="https://github.com/user-attachments/assets/cb5e378b-4adb-4067-9b38-16752f12e36e" />




The input sequence should include the putative Translation Initiation Site (TIS) and the surrounding sequence. The model was trained on HEK293 cell data. The RiboScanner was trained on sequences between 30 bp and 130 bp, so we recommend not exceeding this range. Since most of the training sequences contain only one AUG, we also suggest including only one AUG per input sequence. Additionally, most sequences seen by the model contain 17 nucleotides downstream of the AUG.

# Installation

Optionally, create a new environment for **RiboScanner** :

```
conda create -n RiboScanner python=3.9
conda activate RiboScanner
```

Then, you can install directly from GitHub:

``` pip install git+https://github.com/luciabarb/RiboScanner.git ```

To verify that RiboScanner was installed correctly, run the following command. It should display the help message without errors:

``` RiboScanner --help``` 

# Usage examples

### Predicting leaky scanning

To predict GFP levels associated with leaky scanning for each sequence in a tab-separated dataframe, run:

```sh
RiboScanner predict \
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

| id         | sequence    | length_sequence                  | predictions_GFP   |
-------------|-------------|----------------------------------|-------------------|
| 1          | ATGGAAAG... | 44                               | 14.1292           |
| 2          | ATAAAATA... | 73                               | 0.2097            |
| 3          | 	AGAAGCC... | 73                               | 5.6288            |


## Citation

If you make use of RiboScanner model and/or this pipeline, please cite:


Bram M. P. Verhagen, David Liedtke, Lucía Barbadilla-Martínez, Carlos Alvarado, Valentyn Petrychenko, Michał Świrski, Micha D. Müller, Eivind Valen, Joseph D. Puglisi, Jeroen de Ridder, Niels Fischer and Marvin E. Tanenbaum. "Decoding the sequence requirements for translation initiation." (2026)
