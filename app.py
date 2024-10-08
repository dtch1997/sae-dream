# type: ignore
import os
import streamlit as st
import pandas as pd
import json
import plotly.express as px


# Load the JSON data
@st.cache_data
def load_data():
    with open("dummy_data.json", "r") as f:
        data = json.load(f)
    return data

@st.cache_data
def load_data_from_folder(folder_path='data'):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                feature_data = json.load(f)
                all_data.append(feature_data)
    return all_data

# Load the data
# data = load_data()
data = load_data_from_folder()

# Sidebar
st.sidebar.title("SAE Feature Selection")

# Create a selection box for choosing SAE features
feature_options = [
    f"Layer {d['SAE_metadata']['Layer']} - {d['SAE_metadata']['SAE_type']} - ID {d['SAE_metadata']['Feature_ID']}"
    for d in data
]
selected_feature = st.sidebar.selectbox("Select SAE Feature", feature_options)

# Display the SAE metadata
selected_index = feature_options.index(selected_feature)
metadata = data[selected_index]["SAE_metadata"]

st.sidebar.subheader("SAE Metadata")
for key, value in metadata.items():
    st.sidebar.text(f"{key}: {value}")

# Main dashboard
st.title("SAE Feature Dashboard")

# Prepare data for plotting and table
examples = data[selected_index]["Examples"]
df = pd.DataFrame(examples)

# Plot cross-entropy scores vs EPO metric scores
fig = px.scatter(
    df,
    x="EPO_metric_score",
    y="Cross_entropy_score",
    hover_data=["Tokens"],
    labels={
        "EPO_metric_score": "EPO Metric Score",
        "Cross_entropy_score": "Cross-Entropy Score",
    },
)
st.plotly_chart(fig)

# Table of all examples
st.subheader("Examples Table")


# Function to format the Tokens and Activations columns
def format_list_column(column):
    return column.apply(lambda x: ", ".join(map(str, x)))


df["Tokens"] = format_list_column(df["Tokens"])
df["Activations_per_token"] = format_list_column(df["Activations_per_token"])

st.dataframe(df)

# Instructions for running the app
st.sidebar.markdown("""
## How to run this app:
1. Install required libraries: `streamlit`, `pandas`, `plotly`
2. Ensure `data` folder contains JSON files with feature data
3. Run the command: `streamlit run app.py`
""")
