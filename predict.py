import json
from get_input_args_predict import get_input_args_predict, check_command_line_arguments
from prediction_model import prediction_model
from image_processing import process_image, imshow
from load_model import load_model
from tabulate import tabulate


def main():

    in_arg = get_input_args_predict()
    check_command_line_arguments(in_arg)
    
    
    model = load_model(in_arg.loadstate)
    image_path = in_arg.image_path
    image_processed = process_image(image_path)
    print('\n')

    probs, classes = prediction_model(image_path, model, in_arg.top_k, in_arg.gpu)
    
    
    if in_arg.category_names == None:
     
        data = zip(classes, probs)
        headers = ["Index", "Probability"]
        
        data_table = tabulate(data, headers=headers, tablefmt="grid")
        print(data_table)
    
        
    elif in_arg.category_names == "cat_to_name.json":
 
        flower_names = []
        
        with open(in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        for flower in classes:
            flower_name = cat_to_name[flower]
            flower_names.append(flower_name)

        data = zip(flower_names, probs)
        headers = ["Name", "Probability (%)"]
        
        data_table = tabulate(data, headers=headers, tablefmt="grid")
        print(data_table)
        
    
if __name__ == "__main__":
    main()