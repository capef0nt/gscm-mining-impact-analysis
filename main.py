from src.preprocessing.construct_scores import build_site_construct_table
import pandas as pd

df = pd.read_csv("data/examples/survey_example.csv")
print(build_site_construct_table(df))
