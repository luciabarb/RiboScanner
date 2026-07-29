#!/usr/bin/env python3

import argparse
from .rrwick_help_formatter import MyParser, MyHelpFormatter
from .predict_model import predict_from_dataframe, predict_from_fasta
from .train_model import call_main
from .version import __version__
from .misc import check_cuda
import warnings
import os
import sys
warnings.filterwarnings("ignore")


def main():
    global description
    description = (
        """

                                                        
                                                        
 
    Translation Initiation Sites Model
    Version: """
            + __version__
            + """
    """
    )

    # Main parser ========================================================================
    # ====================================================================================
    parser = MyParser(
        description="R|" + description, formatter_class=MyHelpFormatter, add_help=False
    )
    subparsers = parser.add_subparsers(dest="subparser_name", title="Tasks")

    # Train task =========================================================================
    # ====================================================================================
    train_subparser(subparsers)

    # Predict task =======================================================================
    # ====================================================================================
    predict_subparser(subparsers)


    other_args = parser.add_argument_group("Other")
    other_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    other_args.add_argument(
        "--version",
        action="version",
        version="RiboScanner v" + __version__,
        help="Show program's version number and exit",
    )
    args = parser.parse_args()

    if "func" in args:
        args.func(args)
        sys.stdout = sys.__stdout__
        print(bye_message(), flush=True)
    else:
        parser.print_help()
        exit(1)


def print_arguments(left, right, total_width=80):
    left_width = len(left)
    right_width = total_width - left_width
    right_str = ", ".join(map(str, right)) if isinstance(right, list) else str(right)
    print("{0}: {1:>{2}}".format(left, right_str, right_width - 2))



# Train task ===================================================================

def train_subparser(subparsers):
    "Parses inputs from commandline and returns them as a Namespace object."

    group = subparsers.add_parser(
        "train",
        help="Train a new RiboScanner model from pre-processed data",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description="R|" + description,
    )

    required_args = group.add_argument_group("Required arguments")
    # Arguments for the input files
    required_args.add_argument(  "--input_data", required=True, nargs="+",
        help='Several paths with dataframe files to the train the model. Provide the paths as a space-separated list.'
             'Each file provided will be used for validation and the remaining for training. So there will be as many models as files provided. Supported formats are .csv, .tsv, .xlsx.')

    required_args.add_argument("--output_folder", required=True, type=str,  help="Path to the directory to store all the output files.")
    

    required_args.add_argument('--column_labels', required=True, type = str, help = 'Which column in the input file contains the measurement data. ')

    required_args.add_argument('--column_sequences', required=True, type = str,  help = 'Which column in the input file contains the sequences.')


    ###########
    #Arguments for the model

    model_args = group.add_argument_group("Advanced arguments (for model training)")


    
    model_args.add_argument('--model_architecture', type = str, default = 'MTtrans', help = 'Model architecture to use. (default: MTtrans)',
                            choices = ['MTtrans', 'dense_layers', 'GemoRNA', 'framepool'])
    
    model_args.add_argument('--model_input', type = str, default = None, help = 'Path to an existing model to continue training from. If not given, a new model will be trained from scratch. (default: None)')

    model_args.add_argument('--lr', type = float, default = 0.0005, help = f'Learning rate (default: 0.0005)')

    model_args.add_argument('--batch_size', type = int, default = 32, help = f'Batch size during training (default: 32)')
    
    model_args.add_argument('--num_workers', type = int, default = 1, help = 'Number of workers to use for data loading (default: 1)')

    model_args.add_argument('--epochs', type = int, default = 25, help = 'Number of epochs (default: 25)')
    
    model_args.add_argument('--gradient_clipping', type = float, default = False, help = 'Gradient clipping value, if False, no gradient clipping will be used (default: False)')
    
    model_args.add_argument('--betas',  type=float, nargs='+', default = [0.05,0.05], help = 'Regularization terms, L1 and L2 respectively (default: [0.05,0.05])')
    
    model_args.add_argument('--criterion', type = str, default = 'mse', help = 'Criterion for the loss function (default: mse). Choose from mse or poisson', choices = ['mse', 'poisson'])
    
    model_args.add_argument('--scheduler', type=bool, default=False, help = 'Use scheduler for the learning that changes lr (default: False)')

    ###########
    # Arguments for the sequence treatment
    sequence_args = group.add_argument_group("Other")
 
    sequence_args.add_argument('--type_padding', type = str, default = 'random', help = 'Type of padding, possibilities are right, left, middle, and random (default: random)',
                        choices = ['right', 'left', 'middle', 'random'])
    
    sequence_args.add_argument('--padding_value', type=int, default = 0, help = 'Value for padding (default: 0)')

    sequence_args.add_argument('--padding_with_sequence', type = bool, default = False, help = 'If True, the padding will be with a random sequence, otherwise with a value'
                                                                                ' provided in padding_value, value or random. Bool value, provide True or False (default: False)')

    sequence_args.add_argument('--L_max', type = int, default = 156, help = 'Max length of the sequence (default: 156)')

    sequence_args.add_argument('--adaptors', type=str, nargs='+', default=['AGTGAACC', 'GGCGGCAG'], help='adaptors sequences, several can be given separated by space (default: [AGTGAACC, GGCGGCAG])')
    
    group.set_defaults(func=train)


def train(args):
    # Implement the logic for the train command here
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Train"))
    print("-" * 80)
    print_arguments("Input", args.input_data)
    print_arguments("Output folder", args.output_folder)
    print_arguments("Column with the labels", args.column_labels)
    print_arguments("Column with the sequences", args.column_sequences)
    print_arguments("Model architecture", args.model_architecture)
    print_arguments("Path to existing model to continue training from", args.model_input)
    print_arguments("Learning rate", args.lr)
    print_arguments("Batch size", args.batch_size)
    print_arguments("Number of workers", args.num_workers)
    print_arguments("Number of epochs", args.epochs)
    print_arguments("Gradient clipping", args.gradient_clipping)
    print_arguments("Regularization terms (L1 and L2)", args.betas)
    print_arguments("Criterion for the loss function", args.criterion)
    print_arguments("Use scheduler for the learning that changes lr", args.scheduler)
    print_arguments("Type of padding", args.type_padding)
    print_arguments("Value for padding", args.padding_value)
    print_arguments("Padding with random sequence?", args.padding_with_sequence)
    print_arguments("Max length of the sequence", args.L_max)
    print_arguments("Adaptors sequences", args.adaptors)
    # Same but now filling the output with spaces so it gets 80 characters

    print("=" * 80)
    call_main(args)


# Predict task =================================================================
def predict_subparser(subparsers):

    group = subparsers.add_parser(
        "predict",
        help="Predict promoter activity of sequences in data frame or fasta file using a trained RiboScanner model. "
        "model. The output is a tab-separated file with the sequence and the "
        "predicted score.",
        formatter_class=MyHelpFormatter,
        add_help=False,
        description="R|" + description,
    )

    required_args = group.add_argument_group("Required arguments")

    default_models = [
        os.path.join(os.path.dirname(__file__), "pretrained_models", f"model_fold{i}_final.pth")
        for i in range(10)
    ]

    required_args.add_argument( "--model", nargs="+", 
        default=default_models,
        help="Path(s) to the directory of the model. If you want to perform predictions "
        "for the pre-trained RiboScanner_models, for instance, this should be "
        "pre_trained_models/RiboScanner_models/. If you have trained your own model, "
        "you should pass the path to the directory where the .pth files are stored. ",
    )

    required_args.add_argument( "--input", required=True, help="Path to the input fasta file or dataframe file that must contained a column with the sequences to be predicted." )

    required_args.add_argument( "--column_sequence", default='sequence', type=str, help="Column name in the dataframe that contains the sequences to be predicted. "
        "(default: sequence)."
    )

    required_args.add_argument( "--output", required=True, help="Path to the output file where the predictions will be saved. Output is a "
        "tab-separated file with the sequence, header, and the predicted score."
    )

    required_args.add_argument(  "--n_seqs_per_batch", type=int, default=1,
        help=" Number of sequences to predict simultaneously, increase only if your memory allows it. (Default: 1)"
    )

    required_args.add_argument(
        "--header_only",  action = argparse.BooleanOptionalAction, default=False,
        help="If this flag is set, the output file will not contain the sequences of the\n"
                " input fasta. By default, RiboScanner model shows both the sequence and the header. (Default: False)"
    )
    # ── Model type ────────────────────────────────────────────────────────────
    model_args = group.add_argument_group(
        "Model type",
        description=(
            "Select which model to run. The default is the original RiboScanner "
            "MTtrans model. Use 'noderer' for the Noderer et al. 2014 PWM baseline "
            "or 'evo2' for a trained Evo2 regression checkpoint."
        )
    )

    model_args.add_argument("--model_type", default="MTtrans",
        choices=["MTtrans", "noderer", "evo2", 'dense_layers', 'GemoRNA', 'framepool'],
        help=(
            "Which model architecture to use for prediction. "
            "'MTtrans' = original RiboScanner CNN+GRU (default). "
            "'noderer' = Noderer et al. 2014 dinucleotide PWM (requires a .npz coefficients file). "
            "'evo2'    = Evo2 fine-tuned regression model (requires a best_model.pth checkpoint)."
            "'dense_layers' = Dense layers model. "
            "'GemoRNA' = GemoRNA model. "
            "'framepool' = Framepool model. "
        ))

    # ── Noderer-specific arguments ────────────────────────────────────────────
    noderer_args = group.add_argument_group(
        "Noderer PWM options (only used when --model_type noderer)"
    )

    noderer_args.add_argument("--noderer_order", default="di", choices=["mono", "di"],
        help=(
            "Which PWM order to use for prediction. Must match the .npz file provided "
            "to --model (i.e. if you pass di_pwm_coefficients.npz, use 'di'). "
            "(default: di)"
        ))

    noderer_args.add_argument("--noderer_aug_col", default=None, type=str,
        help=(
            "Column in your input dataframe giving the 0-based index of the 'A' in AUG "
            "for each sequence. If not provided, the first ATG in each sequence is used "
            "automatically. Ignored for FASTA input."
        ))

    noderer_args.add_argument("--noderer_positions", type=int, nargs="+", default=None,
        help=(
            "Which TIS positions were modelled when the PWM was fit "
            "(must match what was used in noderer_pwm.py --positions). "
            "Default: paper's original -6 -5 -4 -3 -2 -1 4 5."
        ))

    # ── Evo2-specific arguments ───────────────────────────────────────────────
    evo2_args = group.add_argument_group(
        "Evo2 options (only used when --model_type evo2)"
    )

    evo2_args.add_argument("--evo2_batch_size", type=int, default=32,
        help=(
            "Number of sequences per Evo2 forward pass. Reduce if you get "
            "CUDA out-of-memory errors. (default: 32)"
        ))

    evo2_args.add_argument("--evo2_regression_path", default=None, type=str,
        help=(
            "Path to evo2_regression.py if it is not on sys.path or installed as part "
            "of RiboScanner. Leave unset if evo2_regression.py lives in the same "
            "directory as RiboScanner/ or is importable as a module."
        ))

    # ── Shared optional arguments ─────────────────────────────────────────────
    shared_args = group.add_argument_group(
        "Shared options"
    )

    shared_args.add_argument("--label_scaler_path", default=None, type=str,
        help=(
            "Path to label_scaler.json saved by noderer_pwm.py or evo2_regression.py "
            "when --normalize_labels was used during training. If provided, predictions "
            "are automatically inverse-transformed back to original label units."
        ))
    ##########
    other_args = group.add_argument_group("Other")


    other_args.add_argument( "--L_max",  
                                default=156, type=int, help="Max length of the sequence (default: 156)")


    other_args.add_argument('--adaptors', 
                                type=str, nargs='+', default=['AGTGAACC', 'GGCGGCAG'], help='adaptors sequences, several can be given separated by space (default: [AGTGAACC, GGCGGCAG])')
    
    other_args.add_argument('--measurement_column', default=False, type=str,
                                help = 'If specified, must be a column name in the input dataframe. The input dataframe must contain a column with the measurements to compare with the predictions. The r2 and scatter plots')

    other_args.add_argument('--split_on_variable', default=False, type=str,
                                help = 'If specified, must be a column name in the input dataframe. The input dataframe will be split into several dataframes based on the values of the variable column to report the r2 and scatter plots. The predictions will be performed for each dataframe separately. (default: False)')

    other_args.add_argument('--type_padding', type = str, default = 'random', help = 'Type of padding, possibilities are right, left, middle, and random (default: random)',
                            choices = ['right', 'left', 'middle', 'random'])
    
    other_args.add_argument('--padding_with_sequence', type = bool, default = False, help = 'If True, the padding will be with a random sequence, otherwise with a value'
                                                                                    ' provided in padding_value, value or random. Bool value, provide True or False (default: False)')

    other_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    other_args.add_argument(
        "--version",
        action="version",
        version="RiboScanner model v" + __version__,
        help="Show program's version number and exit",
    )

    other_args.add_argument(
        "--verbose",
        action="store_true",
        help="If set, the program will print more information about the progress of the predictions.",
    )

    group.set_defaults(func=predict)


def predict(args):
    # Implement the logic for the predict command here
    print(description)
    print("=" * 80)
    print("{: ^80}".format("Predict"))
    print("-" * 80)
    print_arguments("Model", args.model)
    print_arguments("Input", args.input)
    print_arguments("Column sequence in dataframe (if file is not FASTA)", args.column_sequence)
    print_arguments("Output", args.output)
    print_arguments("Number of batches", args.n_seqs_per_batch)
    print_arguments("Include sequence in the output file?", not args.header_only)
    print_arguments("Max length of the sequence", args.L_max)
    print_arguments("Type of padding", args.type_padding)
    print_arguments("Padding with random sequence?", args.padding_with_sequence)
    print_arguments("Adaptors sequences", args.adaptors)
    print_arguments("Model architecture", args.model_type)
    print_arguments("Column with the measurements in the input dataframe", args.measurement_column)
    print_arguments("Split on variable column", args.split_on_variable)
    print_arguments("Verbose", args.verbose)
    if args.model_type == 'noderer':
        print_arguments("Noderer PWM order", args.noderer_order)
        print_arguments("Noderer AUG column", args.noderer_aug_col)
        print_arguments("Noderer positions", args.noderer_positions or "default (-6..-1,+4,+5)")
    if args.model_type == 'evo2':
        print_arguments("Evo2 batch size", args.evo2_batch_size)
        print_arguments("Evo2 regression path", args.evo2_regression_path)
    if args.label_scaler_path:
        print_arguments("Label scaler path", args.label_scaler_path)

    # Same but now filling the output with spaces so it gets 80 characters
    print("=" * 80)

    # Determine GPU/CPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Check if the args.input finishes with .fasta or .fasta.gz
    if args.input.endswith(".fasta") or args.input.endswith(".fa") or args.input.endswith(".fasta.gz") or args.input.endswith(".fa.gz"):
        input_type = "fasta"
        print(f' Detected input file type: {input_type}. If this is incorrect, please check your input file extension.', flush=True)
    else:
        input_type = "dataframe"
    
    print(f' Detected input file type: {input_type}. If this is incorrect, please check your input file extension.', flush=True)

    if input_type == "fasta":
        predict_model = predict_from_fasta(
        input_file=args.input,
        models=args.model,
        batch_size=args.n_seqs_per_batch,
        header_only= args.header_only,
        output_file=args.output,
        L_max=args.L_max,
        adaptors=args.adaptors,
        model_type=args.model_type,
        split_on_variable=args.split_on_variable,
        verbose = args.verbose,
        noderer_order=args.noderer_order,
        noderer_positions=args.noderer_positions,
        label_scaler_path=args.label_scaler_path,
        evo2_batch_size=args.evo2_batch_size,
        evo2_regression_path=args.evo2_regression_path,
        device=device,
        padding=args.type_padding,
        padding_with_sequence=args.padding_with_sequence
        )

    else:
        predict_model = predict_from_dataframe(
        input_file=args.input,
        models=args.model,
        column_sequences=args.column_sequence,
        batch_size=args.n_seqs_per_batch,
        header_only= args.header_only,
        output_file=args.output,
        L_max=args.L_max,
        adaptors=args.adaptors,
        model_type=args.model_type,
        split_on_variable=args.split_on_variable,
        measurement_column=args.measurement_column,
        verbose = args.verbose,
        noderer_aug_col=args.noderer_aug_col,
        noderer_positions=args.noderer_positions,
        label_scaler_path=args.label_scaler_path,
        evo2_batch_size=args.evo2_batch_size,
        evo2_regression_path=args.evo2_regression_path,
        device=device,
        padding=args.type_padding,
        padding_with_sequence=args.padding_with_sequence
        )




####

def bye_message():
    return (
        "\nAll done!\n"
        "If you make use of RiboScanner model in your research, please cite:\n\n"
        "  #### et. al. (2026) \n"
        "   \n"
        "  bioRXiv. ###\n"
        "\n"
        ""
    )



# Main =========================================================================
if __name__ == "__main__":
    main()
