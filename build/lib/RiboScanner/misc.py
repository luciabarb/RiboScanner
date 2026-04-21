from .version import __version__
import argparse

def log(message: str):
    """
    Simple function to print a log message. 
    This writes the message to the console with a version number.
    
    Parameters
    ----------
    message : str
        The message to print to the console.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> log("This is a message")
    [RiboScanner v0.1.0] This is a message
    """
    v = 'RiboScanner v' + __version__
    print(f"[{v}] {message}", flush=True)
    

class check_cuda(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        import torch
        # Show torch info
        print(f"pytorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print("CUDA is available.")
        else:
            print("CUDA is not available.")
        parser.exit()
