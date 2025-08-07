import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from eda_tool.data_loader import load_data
from eda_tool.eda_summary import dataset_summary
from eda_tool.missing_values import plot_missing_values
from eda_tool.visualization import (
    plot_histograms, plot_correlation_matrix, plot_countplots,
    plot_boxplots, plot_violinplots, plot_pairplot, plot_kde
)

st.set_page_config(page_title="Automatic EDA Tool", layout="wide")
st.title("🔍 Automatic EDA Tool")

st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.success("✅ Data loaded successfully")
        selected_columns = st.sidebar.multiselect("Select Columns to Analyze", df.columns, default=df.columns)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Dataset Info", "🧱 Missing Values", "📈 Distributions",
            "🔗 Correlations", "🏷️ Categorical", "⚠️ Outliers",
            "🔀 Relationships", "🌐 KDE Plots"
        ])

        with tab1:
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing (%)", f"{(df.isnull().sum().sum() / df.size) * 100:.2f}%")
            st.write("### Column Info")
            st.write(df[selected_columns].dtypes)
            st.write("### Summary Stats")
            st.write(df[selected_columns].describe())

        with tab2:
            st.subheader("🧱 Missing Values")
            st.write(df[selected_columns].isnull().sum())
            st.pyplot(plot_missing_values(df[selected_columns]))

        with tab3:
            st.subheader("📈 Feature Distributions")
            st.pyplot(plot_histograms(df[selected_columns]))

        with tab4:
            st.subheader("🔗 Correlation Matrix")
            st.pyplot(plot_correlation_matrix(df[selected_columns]))

        with tab5:
            st.subheader("🏷️ Count Plots (Categorical)")
            for fig in plot_countplots(df[selected_columns]):
                st.pyplot(fig)

        with tab6:
            st.subheader("⚠️ Box & Violin Plots")
            for fig in plot_boxplots(df[selected_columns]):
                st.pyplot(fig)
            for fig in plot_violinplots(df[selected_columns]):
                st.pyplot(fig)

        with tab7:
            st.subheader("🔀 Feature Pair Plot")
            with st.spinner("Generating pairplot..."):
                st.pyplot(plot_pairplot(df[selected_columns]))

        with tab8:
            st.subheader("🌐 KDE (Density) Plots")
            for fig in plot_kde(df[selected_columns]):
                st.pyplot(fig)

    else:
        st.error("❌ Failed to load dataset. Please upload a valid CSV file.")
