import pandas as pd
import numpy as np
import os
import shutil

data_folder = 'challengeToFill.csv'
preprocessed_folder = 'PreProcessedData'

num_of_users = 40
num_of_seg = 150
num_of_words_in_seg = 100

def main():
    input_file = os.path.abspath(data_folder)

    df = pd.read_csv(input_file, usecols=range(1,num_of_seg))

    for i in range(0,num_of_users):
        user = df.iloc[i]

if __name__ == '__main__':
    main()