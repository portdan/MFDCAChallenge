import pandas as pd
import numpy as np
import os
import shutil

data_folder = 'FraudedRawData'
preprocessed_folder = 'PreProcessedData'

num_of_users = 40
num_of_seg = 150
num_of_words_in_seg = 100

def main():


    li = []
    users = ['User' + str(i) for i in range(num_of_users)]

    if os.path.exists(os.path.abspath(preprocessed_folder)):
        shutil.rmtree(os.path.abspath(preprocessed_folder))

    for user in users:

        os.makedirs(os.path.abspath(os.path.join(preprocessed_folder, str(user))))

        input_file = os.path.abspath(os.path.join(data_folder, str(user)))

        df = pd.read_csv(input_file, header=None, names=['Vocalbulary'])

        for seg in range(0, num_of_seg):
            start = seg*num_of_words_in_seg

            d = df.iloc[start: start + num_of_words_in_seg]
            d_unique = d.groupby(['Vocalbulary'], as_index=True).size().to_frame()
            d_unique.columns = ['Count']
            d_unique = d_unique.sort_values(by=['Count'], ascending=False)
            d_unique['Vocalbulary'] = list(d_unique.index)
            d_unique = d_unique[['Vocalbulary','Count']]

            output_file = os.path.abspath(os.path.join(preprocessed_folder, str(user), 'seg-' + str(seg) + '.csv'))
            d_unique.to_csv(output_file, index=False)

        d_unique = df.groupby(['Vocalbulary'], as_index=True).size().to_frame()
        d_unique.columns = ['Count']
        d_unique = d_unique.sort_values(by=['Count'], ascending=False)
        d_unique['Vocalbulary'] = list(d_unique.index)
        d_unique = d_unique[['Vocalbulary', 'Count']]

        output_file = os.path.abspath(os.path.join(preprocessed_folder, str(user), 'total.csv'))
        d_unique.to_csv(output_file, index=False)

        li.append(df)

    total = pd.concat(li, axis=0, ignore_index=True)

    unique = total.groupby(['Vocalbulary'], as_index=True).size().to_frame()
    unique.columns = ['Count']
    unique = unique.sort_values(by=['Count'], ascending=False)
    unique['Vocalbulary'] = list(unique.index)
    unique = unique[['Vocalbulary', 'Count']]

    output_file = os.path.abspath(os.path.join(preprocessed_folder, 'total.csv'))
    unique.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()