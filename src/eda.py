import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# PLOTTING STYLE
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


# LOAD DATA
df_before = pd.read_csv("../data/raw/before_cleaning.csv")
df_after = pd.read_csv("../data/processed/after_cleaning.csv")

england = pd.read_csv("../data/processed/england.csv")
age = pd.read_csv("../data/processed/age.csv")
gender = pd.read_csv("../data/processed/gender.csv")
deprivation = pd.read_csv("../data/processed/2015_deprivation_decile.csv")

