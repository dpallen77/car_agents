import pandas as pd

cars = pd.read_csv("cars_dataset.csv")
print(cars.head())        # first 5 rows
print(cars.columns)       # exact column names
print(len(cars), "rows")