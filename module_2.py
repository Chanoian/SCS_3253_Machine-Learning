import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()
lin_reg = LinearRegression()
dec_tree = DecisionTreeRegressor()

HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing_ocean_proximity = housing['ocean_proximity']
housing_ocean_proximity_enc = encoder.fit_transform(housing_ocean_proximity)
housing_ocean_proximity_one_hot_enc = one_hot_encoder.fit_transform(housing_ocean_proximity_enc.reshape(-1,1))

