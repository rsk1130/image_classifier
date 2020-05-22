
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args_train():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. 
      2. 
      3. 
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    
    parser = argparse.ArgumentParser(description="")
    
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    
    parser.add_argument('data_dir', type = str)
    parser.add_argument('--save_dir', type = str, default = 'save_directory')
    parser.add_argument('--arch', type = str, default = 'vgg16')
    parser.add_argument('--epochs', type = int, default = 3)
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--hidden_units', type = int, default = 1024)                    
    parser.add_argument('--gpu', type = str, default = 'cuda')                    
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    
    return parser.parse_args()

                        
                        
def check_command_line_arguments(in_arg):
    """
    For Lab: Classifying Images - 7. Command Line Arguments
    Prints each of the command line arguments passed in as parameter in_arg, 
    assumes you defined all three command line arguments as outlined in 
    '7. Command Line Arguments'
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console  
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\ndata_dir =", in_arg.data_dir, 
              "\nsave_dir =", in_arg.save_dir, "\narch =", in_arg.arch, "\nepochs =", in_arg.epochs,
              "\nlearning_rate =", in_arg.learning_rate, "\nhidden_units =", in_arg.hidden_units,
              "\ngpu =", in_arg.gpu)                        