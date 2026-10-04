import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Iris Dashboard", layout="wide")
st.title("Iris Dataset Interactive Dashboard")


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Series(iris.target).map(dict(enumerate(iris.target_names)))
    return df


df = load_data()
features = list(df.columns[:-1])

# Sidebar filters
st.sidebar.header("Filter Data")
species_options = list(df["species"].unique())
species_filter = st.sidebar.multiselect(
    "Select Species", options=species_options, default=species_options
)

filtered_df = df[df["species"].isin(species_filter)]

if filtered_df.empty:
    st.warning("Select at least one species in the sidebar.")
    st.stop()

# Dataset preview (actual rows)
st.subheader("Dataset Preview")
st.dataframe(filtered_df, width="stretch")

# Summary statistics
st.subheader("Summary Statistics")
st.dataframe(filtered_df.describe(), width="stretch")

# Scatter plot
st.subheader("Scatter Plot")
c1, c2 = st.columns(2)
x_axis = c1.selectbox("X-axis", features, index=0)
y_axis = c2.selectbox("Y-axis", features, index=1)

fig_scatter = px.scatter(
    filtered_df, x=x_axis, y=y_axis, color="species",
    title=f"{x_axis} vs {y_axis}",
)
st.plotly_chart(fig_scatter, width="stretch")

# Histogram
st.subheader("Histogram Plot")
feature = st.selectbox("Select Feature", features, key="hist")

fig_hist = px.histogram(
    filtered_df, x=feature, color="species",
    marginal="box", nbins=20,
    title=f"Distribution of {feature}",
)
st.plotly_chart(fig_hist, width="stretch")

# Correlation heatmap
st.subheader("Correlation Heatmap")
corr = filtered_df[features].corr()

fig_heatmap = px.imshow(
    corr, text_auto=".2f",
    color_continuous_scale="Viridis",
    title="Feature Correlation Heatmap",
)
st.plotly_chart(fig_heatmap, width="stretch")

# Pair plot
st.subheader("Pair Plot")
fig_pair = px.scatter_matrix(
    filtered_df, dimensions=features, color="species",
    title="Pairwise Feature Relationships",
)
fig_pair.update_traces(diagonal_visible=False)
fig_pair.update_layout(height=800)
st.plotly_chart(fig_pair, width="stretch")

st.markdown("---")
st.markdown("Built using Streamlit")