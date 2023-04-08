import pandas as pd
import numpy as np


def get_count_of_ones_and_twos(predict):
    print("Number of predicted ones",np.count_nonzero(predict==1))
    print("Number of predicted twos",np.count_nonzero(predict==2))
    
def create_submission(predict,filename):
    sub_file = pd.read_csv("./data/sample_submission.csv")
    sub_file["Predicted"] = predict
    sub_file.to_csv(filename,index=False)
    print(filename," Created")