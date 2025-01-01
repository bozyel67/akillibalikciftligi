from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import joblib
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv("realfishdataset.csv")


#AYKIRI DEGERLERIN TESPITI (1. CALISMA)
lower_limit_ph = np.percentile(train_df["ph"], 25) - \
                (1.5 * (np.percentile(train_df["ph"], 75) - \
                        np.percentile(train_df["ph"], 25)))

lower_limit_temperature = np.percentile(train_df["temperature"], 25) - \
                        (1.5 * (np.percentile(train_df["temperature"], 75) - \
                                np.percentile(train_df["temperature"], 25)))

lower_limit_turbidity = np.percentile(train_df["turbidity"], 25) - \
                        (1.5 * (np.percentile(train_df["turbidity"], 75) - \
                                np.percentile(train_df["turbidity"], 25)))

upper_limit_ph = np.percentile(train_df["ph"], 75) + \
                 (1.5 * (np.percentile(train_df["ph"], 75) - \
                         np.percentile(train_df["ph"], 25)))

upper_limit_turbidity = np.percentile(train_df["turbidity"], 75) + \
                        (1.5 * (np.percentile(train_df["turbidity"], 75) - \
                                np.percentile(train_df["turbidity"], 25)))

upper_limit_temperature = np.percentile(train_df["temperature"], 75) + \
                          (1.5 * (np.percentile(train_df["temperature"], 75) - \
                                  np.percentile(train_df["temperature"], 25)))


#AYKIRI DEGERLERIN VERI SETINDEN CIKARTILMASI (1. CALISMA)
outliers_ph = train_df.loc[(train_df["ph"] < lower_limit_ph) | (train_df["ph"] > upper_limit_ph)]
outliers_turbidity = train_df.loc[(train_df["turbidity"] < lower_limit_turbidity) | (train_df["turbidity"] > upper_limit_turbidity)]
outliers_temperature = train_df.loc[(train_df["temperature"] < lower_limit_temperature) | (train_df["temperature"] > upper_limit_temperature)]

original_count = len(train_df)
train_df = train_df.loc[
    (train_df["ph"] >= lower_limit_ph) & 
    (train_df["ph"] <= upper_limit_ph) &
    (train_df["temperature"] >= lower_limit_temperature) & 
    (train_df["temperature"] <= upper_limit_temperature) &
    (train_df["turbidity"] >= lower_limit_turbidity) & 
    (train_df["turbidity"] <= upper_limit_turbidity)]


#AYKIRI DEGERLERIN TESPITI (2. CALISMA)
def calculate_lower_limits(df, variable):
    lower_limits = {}
    for fish_type in df["fish"].unique():
        species_data = df[df["fish"] == fish_type]
        lower_limit = np.percentile(species_data[variable], 25) - 1.5 * (
            np.percentile(species_data[variable], 75) - np.percentile(species_data[variable], 25)
        )
        lower_limits[fish_type] = lower_limit
    return lower_limits

ph_lower_limits = calculate_lower_limits(train_df, "ph")
turbidity_lower_limits = calculate_lower_limits(train_df, "turbidity")
temperature_lower_limits = calculate_lower_limits(train_df, "temperature")

def calculate_upper_limits(df, variable):
    upper_limits = {}
    for fish_type in df["fish"].unique():
        species_data = df[df["fish"] == fish_type]
        upper_limit = np.percentile(species_data[variable], 75) + 1.5 * (
            np.percentile(species_data[variable], 75) - np.percentile(species_data[variable], 25)
        )
        upper_limits[fish_type] = upper_limit
    return upper_limits

ph_upper_limits = calculate_upper_limits(train_df, "ph")
turbidity_upper_limits = calculate_upper_limits(train_df, "turbidity")
temperature_upper_limits = calculate_upper_limits(train_df, "temperature")

#AYKIRI DEGERLERIN VERI SETINDEN CIKARTILMASI (2. CALISMA)
def is_outlier(row, variable, lower_limits, upper_limits):
    return (row[variable] < lower_limits[row["fish"]]) or (row[variable] > upper_limits[row["fish"]])

outliers_ph = train_df[train_df.apply(lambda row: is_outlier(row, "ph", ph_lower_limits, ph_upper_limits), axis=1)]
outliers_turbidity = train_df[train_df.apply(lambda row: is_outlier(row, "turbidity", turbidity_lower_limits, turbidity_upper_limits), axis=1)]
outliers_temperature = train_df[train_df.apply(lambda row: is_outlier(row, "temperature", temperature_lower_limits, temperature_upper_limits), axis=1)]

original_count = len(train_df)
train_df = train_df[
    ~train_df.apply(lambda row: is_outlier(row, "ph", ph_lower_limits, ph_upper_limits), axis=1) &
    ~train_df.apply(lambda row: is_outlier(row, "turbidity", turbidity_lower_limits, turbidity_upper_limits), axis=1) &
    ~train_df.apply(lambda row: is_outlier(row, "temperature", temperature_lower_limits, temperature_upper_limits), axis=1)
]


# VERİ MODELLEME
X = train_df[["ph", "turbidity", "temperature"]]
y = train_df["fish"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rf_clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    min_samples_split=5,
    random_state=42
)
rf_clf.fit(X_train, y_train)

# Modeli .pkl dosyasına kaydet
joblib.dump(rf_clf, "veri_modeli_son.pkl")
print("Model basari ile kaydedildi!")