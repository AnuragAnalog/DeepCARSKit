#!/usr/bin/python3

import numpy as np
import pandas as pd

from zipfile import ZipFile

# Make sure the col_dict has three mandatory columns
# user_id, item_id, rating
# Except for the rating, all the other columns are string type

def convert_data(data, col_dict):
    """
    Convert the data frame into the form of the deepcarskit input
    """

    # Convert the rating to the float type
    data[col_dict["rating"]] = data[col_dict["rating"]].astype(float)

    # Convert the string columns into the object type
    for col in data.columns:
        if col != col_dict["rating"]:
            data[col] = data[col].astype('str')

    all_cats = [col_dict["user_id"], col_dict["item_id"], col_dict["rating"]]
    print(data.info())
    data["context"] = data.drop(all_cats, axis=1).apply(lambda x: '_'.join(x), axis=1)
    data["uc_id"] = data["context"] + "_" + data[col_dict["user_id"]]

    new_cols = list()
    for col in data.columns:
        if col != col_dict["rating"]:
            new_cols.append(col+":token")
        else:
            new_cols.append(col+":float")
    data.columns = new_cols

    return data

if __name__ == "__main__":
    with ZipFile('./dataset/Mobile_Frappe.zip', 'r') as zipObj:
        zipObj.extractall("./dataset/")

    data = pd.read_csv('./dataset/Mobile_Frappe/frappe/frappe.csv', sep="\t")
    col_dict = {"user_id": "user", "item_id": "item", "rating": "cnt"}
    data = convert_data(data, col_dict)
    print(data.head())