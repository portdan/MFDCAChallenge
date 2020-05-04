import pandas as pd
import os

def main():

    li = []
    filenames = [os.path.abspath(os.path.join('FraudedRawData', 'User' + str(i))) for i in range(40)]

    for file in filenames:
        data = pd.read_csv(file, header=None)
        li.append(data)

    frame = pd.concat(li, axis=0, ignore_index=True)

    frame.columns = ['Vocalbulary']

    frame2 = frame.drop_duplicates()

    #frame2 = frame.groupby('Vocalbulary').nunique()

    frame2.to_csv('bla.csv', index=False)

if __name__ == '__main__':
    main()