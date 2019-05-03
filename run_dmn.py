import numpy as np 
import pandas as pd 
import argparse
import glob 
from dmn_model_glove import DMNGlovemodelRun


# run experiments by setting the different arguments
# otherwise runs default settings
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', help="Location of the data folder with the train/test folders for each task", default="../babi_dataset/")
    parser.add_argument('-bs', help="Batch size to use", default='64')
    parser.add_argument('-ep', help='Number of Epochs', default='50')
    parser.add_argument('-ug', help="If gloVe should be used (0 is False)", default=0)
    parser.add_argument('-oa', help="If accuracy output at every epoch (0 is False)" , default=0)
    parser.add_argument('-qt', help="Which bAbI task to run", default='1')
    parser.add_argument('-hs', help="Hidden layers size" , default='400')
    parser.add_argument('-eb', help="GloVe Embedding Size" , default='200')

    args = vars(parser.parse_args())

    path_to_data = args['dp']
    babi_task_num = str(args['qt'])
    use_glove = int(args['ug'])
    epoch_num = int(args['ep'])
    batch_size  = int(args['bs'])
    hiddensize  = int(args['hs'])
    out_acc = int(args['oa'])
    ebsize = int(args['eb'])

    train_path = path_to_data + 'train/qa' + babi_task_num + '_train.txt'
    test_path = path_to_data + 'test/qa' + babi_task_num + '_valid.txt'
    
    if use_glove < 1:
        #DMNmodelRun(train_path, test_path, babi_task_num,batch_size, epoch_num, use_glove, hiddensize, out_acc)
    else:
        DMNGlovemodelRun(train_path, test_path, babi_task_num,batch_size, epoch_num, use_glove, hiddensize, out_acc, ebsize)
    
if __name__ == "__main__":
    main()

