import os
import re
import sys
import pandas as pd
import numpy as np
import pickle
def get_patients_id(raw_dir):
    images = []
    num_patients = 0
    patients_id_num = {}
    patients_db = pd.DataFrame(columns = ['a', 'ss', 'gs', 'nos', 'cc', 'fcc', 'fc'])
    for image_file in os.listdir(raw_dir):
        patient_id = image_file.split('_')[0]
        label = image_file.split('_')[-1].split('.')[0]
        if patient_id not in patients_db.index:
            patients_db.loc[patient_id] = [0,0,0,0,0,0,0]
            patients_db.loc[patient_id][label]+=1
        else:
            patients_db.loc[patient_id][label] += 1
    return patients_db
if __name__ == '__main__':
    raw_dir = sys.argv[1]
    stat_info_save_dir = sys.argv[2]
    patients_db = get_patients_id(raw_dir)
    print(list(patients_db.index))
    with open(stat_info_save_dir, 'wb') as f:
        pickle.dump(list(patients_db.index), f)