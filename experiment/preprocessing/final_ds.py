from preprocessing import preprocessings, name
import pandas as pd

housing = pd.read_csv("files/housing.csv")

final_dss = preprocessings.fit_transform(housing)

print(name)