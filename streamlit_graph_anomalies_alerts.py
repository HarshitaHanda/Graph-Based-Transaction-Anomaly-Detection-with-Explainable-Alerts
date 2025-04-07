
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("transaction_anomalies_dataset.csv")
    df["Time_of_Day"] = pd.to_datetime(df["Time_of_Day"], format="%H:%M").dt.hour
    df["Day_of_Week"] = df["Day_of_Week"].astype("category").cat.codes
    df["Gender"] = df["Gender"].astype("category").cat.codes
    df["Account_Type"] = df["Account_Type"].astype("category").cat.codes
    np.random.seed(42)
    df["User_ID"] = np.random.randint(1000, 1100, size=len(df))
    df["To_User_ID"] = np.random.choice(df["User_ID"], size=len(df))
    return df

df = load_data()

# Build Graph
G = nx.Graph()
user_features = df.groupby("User_ID").agg({
    "Transaction_Amount": "mean",
    "Transaction_Volume": "mean",
    "Frequency_of_Transactions": "mean",
    "Income": "mean",
    "Gender": "first",
    "Age": "mean"
}).reset_index()

for _, row in user_features.iterrows():
    G.add_node(row["User_ID"],
               avg_amount=row["Transaction_Amount"],
               avg_volume=row["Transaction_Volume"],
               freq=row["Frequency_of_Transactions"],
               income=row["Income"],
               gender=row["Gender"],
               age=row["Age"])

for _, row in df.iterrows():
    G.add_edge(row["User_ID"], row["To_User_ID"],
               amount=row["Transaction_Amount"],
               time=row["Time_of_Day"],
               day=row["Day_of_Week"])

# Compute anomaly features
anomaly_df = pd.DataFrame(index=G.nodes())
anomaly_df["degree"] = pd.Series(dict(G.degree()))
anomaly_df["clustering"] = pd.Series(nx.clustering(G))

avg_edge_amount = {}
for node in G.nodes():
    edges = G.edges(node, data=True)
    amounts = [edata["amount"] for _, _, edata in edges]
    avg_edge_amount[node] = np.mean(amounts) if amounts else 0
anomaly_df["avg_edge_amount"] = pd.Series(avg_edge_amount)

# Normalize and score
scaler = StandardScaler()
scaled = scaler.fit_transform(anomaly_df)
weights = np.array([0.4, 0.3, 0.3])
anomaly_df["anomaly_score"] = scaled @ weights

# Flag anomalies
threshold = np.percentile(anomaly_df["anomaly_score"], 90)
anomaly_df["is_anomaly"] = (anomaly_df["anomaly_score"] >= threshold).astype(int)

# Explainable alerts
def generate_alerts(row):
    base = "⚠️ User {}: ".format(int(row.name))
    alert = []
    if row["avg_edge_amount"] > 1200:
        alert.append(f"High transaction average: ₹{row['avg_edge_amount']:.2f}")
    if row["degree"] > 30:
        alert.append(f"High connectivity (degree = {row['degree']})")
    if row["clustering"] < 0.2:
        alert.append("Low clustering (weak community behavior)")
    return base + ", ".join(alert) if alert else "✅ No unusual behavior"

anomaly_df["alert_message"] = anomaly_df.apply(generate_alerts, axis=1)

# Streamlit Interface
st.title("🔍 Graph-Based Transaction Anomaly Detection")
st.markdown("Detect anomalous users based on behavior & network topology. Built with NetworkX & Streamlit.")

# Distribution Plot
st.subheader("📊 Anomaly Score Distribution")
fig, ax = plt.subplots()
sns.histplot(anomaly_df["anomaly_score"], kde=True, bins=30, ax=ax)
ax.axvline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
ax.set_xlabel("Anomaly Score")
ax.set_title("Anomaly Score Histogram")
ax.legend()
st.pyplot(fig)

# Anomalous Users
st.subheader("🚨 Top Anomalous Users with Explainable Alerts")
top_anomalies = anomaly_df[anomaly_df["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False)
st.dataframe(top_anomalies[["degree", "clustering", "avg_edge_amount", "anomaly_score", "alert_message"]].head(10))

# Score Filter
st.subheader("📌 Explore Users by Anomaly Score")
min_score, max_score = float(anomaly_df["anomaly_score"].min()), float(anomaly_df["anomaly_score"].max())
score_range = st.slider("Select Score Range", min_value=min_score, max_value=max_score, value=(min_score, max_score))
filtered = anomaly_df[(anomaly_df["anomaly_score"] >= score_range[0]) & (anomaly_df["anomaly_score"] <= score_range[1])]
st.dataframe(filtered[["degree", "clustering", "avg_edge_amount", "anomaly_score", "alert_message"]])
