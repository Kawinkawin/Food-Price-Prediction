import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load synthetic data
df = pd.read_csv(r"D:\CLG\SEM-V\BDA\data\synthetic_food_data.csv")

# Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical variables
label_encoder_region = LabelEncoder()
label_encoder_food = LabelEncoder()

df["Region"] = label_encoder_region.fit_transform(df["Region"])
df["Food_Item"] = label_encoder_food.fit_transform(df["Food_Item"])

# Save processed data
df.to_csv("data/processed_food_data.csv", index=False)
print("Data preprocessed and saved.")
