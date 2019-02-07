# State of the Art(Dec.,2018) Bangla Handwritten Character Recognition on 84 classes

This is the code for the paper "Bengali Handwritten Character Classification using Transfer Learning on Deep Convolutional Neural Network" authored by Swagato Chatterjee ( https://github.com/swagato-c/ ), Rwik Kumar Dutta ( https://github.com/rwikdutta ) and others. 

The weight file is given which can easily be loaded using FastAI library and used for further training or inference. For training, the weight file BanglaLekha_weights_train.pth needs to be used. For further inference, the weight file BanglaLekha_weights_infer.pkl needs to be used. The .pkl weight file holds a little bit more information which helps during inference but can't be used for further training ( as of writing this document ).

Using this weights, 96.13% accuracy was achieved on the BanglaLekha-Isolated Dataset ( https://data.mendeley.com/datasets/hf6sf8zrkc/2 ). The train and test set we used is given in the file data_25.csv.


## Installation Steps:
Note: The code was tested on a machine running Ubuntu 18.04.1 LTS. The fastai version this was tested on is 1.0.42 and torch version 1.0.0 with GPU memory of ~11.5 GB. We would recommend setting this up on virtualenv so that it might not interfere with your existing python installations. To know more about virtualenv, visit this link: https://docs.python-guide.org/dev/virtualenvs/
- Download the dataset from this link: https://data.mendeley.com/datasets/hf6sf8zrkc/2
- Run `unzip -q BanglaLekha-Isolated.zip` to extract the zip file. This should create a directory named BanglaLekha-Isolated in your current directory.
- `pip install fastai==1.0.42` ( if you face problems with this step, refer here https://github.com/fastai/fastai/blob/master/README.md#installation )
- `git clone https://github.com/swagato-c/bangla-hwcr-present`
- `mkdir BanglaLekha-Isolated/Images/models`
- `cp model_files_final_weights_train.pth BanglaLekha-Isolated/Images/models/` ( Required for training since the file needs to be present in the directory given using the {--images_path} flag in train.py.)
- `cp data_25.csv BanglaLekha-Isolated/Images/` ( Required for training since the file needs to be present in the directory given using the {--images_path} flag in train.py.)
- See Usage section to run the appropiate command as per your requirement.
- 

## Usage:
To view the detailed description of any file, you can put in `python <filename.py> -h` 
1. For interpretation:
````
usage: interp.py [-h] -f INPUT_FILE -w WEIGHT_FILE [-m] [-o CSV_OUTPUT]
                 [-i IMAGE_PATH] [-c CSV_COLUMN]

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --input_file INPUT_FILE
                        path of file to be interpreted...if used wih -s or
                        --pred_single, then this should be a csv file
                        containing all the file names that has to be predicted
  -w WEIGHT_FILE, --weight_file WEIGHT_FILE
                        Path of Weight File
  -m, --pred_multiple   Predict multiple images...if used,input file should be
                        a csv file containing all the image file paths to be
                        predicted and if not used, then the input_file should
                        be an image file
  -o CSV_OUTPUT, --csv_output CSV_OUTPUT
                        if -m or --pred_multiple is used, then this argument
                        has to be specified containing the output location of
                        the csv file which will contain the predictions
  -i IMAGE_PATH, --image_path IMAGE_PATH
                        if -m or --pred_multiple is used, provide the base
                        path of the images.
  -c CSV_COLUMN, --csv_column CSV_COLUMN
                        if -m or --pred_multiple is used, then the column name
                        of csv_file is mentioned here.
````
For interpreting a single file (sample command):
````
python interp.py --input_file BanglaLekha-Isolated/Images/83/01_0001_0_11_0916_1913_83.png --weight_file model_files_final_weights_interp.pkl
````
Prints a tuple containing:
- Predicted class
- Softmax probability of each of the 83 classes


For interpreting multiple files (sample command):
````
!python interp.py --input_file BanglaLekha-Isolated/Images/data_25_mod.csv --weight_file model_files_final_weights_interp.pkl --pred_multiple --csv_output out_pred.csv --image_path BanglaLekha-Isolated/Images -csv_column x
````
Saves a csv file to the path given in --csv_output parameter. If no path is mentioned, the file is stored in current directory. The csv output file contains the path of each input image, the probabilities for each of the 84 classes and the  predicted class. 

2. For training:
````
usage: train.py [-h] -i INPUT_FILE_NAME -im IMAGES_PATH [-p]
                [-m MODEL_FILE_NAME] [-f] -lrl LOWER_LR [-lrh HIGHER_LR] -sz
                IMAGE_SIZE -bs BATCH_SIZE -e EPOCHS [-s CUSTOM_SAVE_SUFFIX]
                [-ei]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE_NAME, --input_file_name INPUT_FILE_NAME
                        this should be the name of the csv file containing all
                        the file names (col_name: x), the ground truth
                        (col_name: y) and whether each file belongs to test or
                        validation (col_name:is_valid)
  -im IMAGES_PATH, --images_path IMAGES_PATH
                        path of the images folder...the path present in the
                        csv file will be appended to this path for each
                        image...even the output files are stored in this
                        directory
  -p, --pretrained      If --model_file or -m is not used and --pretrained or
                        -p flag is used, then a pretrained weights trained on
                        ImageNet is downloaded and used...if --pretrained or
                        -p is not specified and --model_file or -m is also not
                        mentioned, random initialization is done...if
                        --model_file or -m is mentioned, this value is ignored
  -m MODEL_FILE_NAME, --model_file_name MODEL_FILE_NAME
                        Name of Model File...should be present at
                        --images_path/models/ directory
  -f, --freeze_layers   If used, freezes the non-FC layers
  -lrl LOWER_LR, --lower_lr LOWER_LR
                        Lower learning rate for training...if -f or
                        --freeze_layers flag is used, then -lrh or --higher_lr
                        is not required
  -lrh HIGHER_LR, --higher_lr HIGHER_LR
                        Higher lr for training...only required if -f or
                        --freeze_layers flag is not used
  -sz IMAGE_SIZE, --image_size IMAGE_SIZE
                        Image size to be used for training
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size to be used
  -e EPOCHS, --epochs EPOCHS
                        No of epochs to train for
  -s CUSTOM_SAVE_SUFFIX, --custom_save_suffix CUSTOM_SAVE_SUFFIX
                        Custom suffix to be added at the end of the name of
                        files to be stored as output
  -ei, --export_inference
                        Export model for inference (.pkl file)...the file is
                        saved as export.pkl under the directory given in -im
                        or --images_path/models/ directory
````
For training (sample command):
````
python train.py --input_file_name data_25.csv --images_path BanglaLekha-Isolated/Images/ --model_file model_files_final_weights_train.pth --lower_lr 1e-06 --higher_lr 6e-04 --image_size 128 --batch_size 256 --epochs 1 --custom_save_suffix check_1 --export_inference
````

Trains the model and then saves the confusion matrix in the --images_path directory. The name of the confusion matrix csv file ends in `_confusion_matrix.csv`. The output of the CSVLogger callback ( containing training details epoch wise ) is stored in the file which ends in `_history.csv`. The model is saved in the {--images_path}/models directory  with the extension .pth. This .pth should be used during the further stages of training.

If the flag --export_inference is used, another .pkl file is saved in the {--images_path}/models directory which contains the necessary information for inference. Please note that interp.py only works with this file and can't be used with the .pth file. The reason we can't use the .pkl file for training too is because ( as of writing this article ), the model stored and then loaded from the .pkl file doesn't support differential learning rates, which is necessary for further training.