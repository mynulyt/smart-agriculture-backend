import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load datasets
df_yield = pd.read_csv('datasets/crop_yield.csv')
df_soil = pd.read_csv('datasets/state_soil_data.csv')
df_weather = pd.read_csv('datasets/state_weather_data_1997_2020.csv')
df_recommendation = pd.read_csv('datasets/Crop_recommendation.csv')

# Rename columns
df_yield.rename(columns={'state': 'State'}, inplace=True)
df_soil.rename(columns={'state': 'State'}, inplace=True)
df_weather.rename(columns={'state': 'State'}, inplace=True)

# Normalize text
for df in [df_yield, df_soil, df_weather, df_recommendation]:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower().str.strip()

# Data Fusion
fusion_temp = pd.merge(
    df_yield,
    df_soil,
    on='State',
    how='inner'
)

final_fusion_df = pd.merge(
    fusion_temp,
    df_weather,
    on=['State', 'year'],
    how='inner'
)

# Encoding
encoders = {}
df_processed = final_fusion_df.copy()

for col in ['crop', 'season', 'State']:

    encoders[col] = LabelEncoder()

    df_processed[col] = encoders[col].fit_transform(
        df_processed[col]
    )

# Classification Model
X_class = df_recommendation.drop(
    'label',
    axis=1
)

y_class = df_recommendation['label']

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_class, y_class)

# Regression Model
X_reg = df_processed.drop(
    ['yield', 'production'],
    axis=1
)

y_reg = df_processed['yield']

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(X_reg, y_reg)

# Save Models
pickle.dump(
    classifier,
    open('models/classifier.pkl', 'wb')
)

pickle.dump(
    regressor,
    open('models/regressor.pkl', 'wb')
)

pickle.dump(
    encoders,
    open('models/encoders.pkl', 'wb')
)

print("Models Saved Successfully")