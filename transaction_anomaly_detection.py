
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("transaction_anomalies_dataset.csv")

# Encode categorical features
categorical_cols = ["Day_of_Week", "Time_of_Day", "Gender", "Account_Type"]
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Select relevant features
features = [
    "Transaction_Amount", "Transaction_Volume", "Average_Transaction_Amount",
    "Frequency_of_Transactions", "Time_Since_Last_Transaction", "Day_of_Week",
    "Time_of_Day", "Age", "Gender", "Income", "Account_Type"
]

X = df[features]

# Train Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df["Anomaly_Score"] = model.decision_function(X)
df["Anomaly"] = model.predict(X)

# Map anomalies: 1 -> normal, -1 -> anomaly
df["Anomaly"] = df["Anomaly"].map({1: 0, -1: 1})

# Summary stats
print("Total Anomalies Detected:", df["Anomaly"].sum())
print("\nAnomalies by Account Type:")
print(df[df["Anomaly"] == 1]["Account_Type"].value_counts())

# Plot anomalies
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df["Transaction_Amount"], y=df["Average_Transaction_Amount"],
    hue=df["Anomaly"], palette={0: "blue", 1: "red"}, alpha=0.6
)
plt.title("Anomaly Detection in Transactions")
plt.xlabel("Transaction Amount")
plt.ylabel("Average Transaction Amount")
plt.legend(title="Anomaly")
plt.tight_layout()
plt.show()
