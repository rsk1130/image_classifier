import argparse


def get_input_args_predict():
  
    
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('image_path', type = str)
    parser.add_argument('loadstate', type = str)
    parser.add_argument('--top_k', type = int, default = 1)
    parser.add_argument('--category_names', type = str, default = None)                 
    parser.add_argument('--gpu', type = str, default = 'cuda')                    
    
    return parser.parse_args()

                        
def check_command_line_arguments(in_arg):
    
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\nimage_path =", in_arg.image_path, 
              "\nloadstate =", in_arg.loadstate, "\nTop K =", in_arg.top_k, "\nMappings =", in_arg.category_names,
              "\ngpu =", in_arg.gpu) 