from sklearn.datasets import load_iris
import pandas as pd 

iris = load_iris(as_frame=True, return_X_y=True)

df = iris[0]
target = iris[1]

df["species"] = target

df.to_csv("iris.csv")
#print(target.head())