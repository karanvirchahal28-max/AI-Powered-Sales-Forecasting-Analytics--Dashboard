import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide"
)
tab1, tab2, tab3 = st.tabs(
    ["Overview", "Products", "Customers"]
)
# ==================================
# Load Dataset
# ==================================
df = pd.read_csv("data/sales_data.csv")
df = pd.read_csv("data/sales_data.csv")

full_df = df.copy()
# ==================================
# SIDEBAR FILTERS
# ==================================

st.sidebar.header("Filters")

# Region Filter
selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + sorted(df["Region"].dropna().unique())
)

# Category Filter
selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["Category"].dropna().unique())
)

# State Filter
selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(df["State/Province"].dropna().unique())
)

# Apply Filters
if selected_region != "All":
    df = df[df["Region"] == selected_region]

if selected_category != "All":
    df = df[df["Category"] == selected_category]

if selected_state != "All":
    df = df[df["State/Province"] == selected_state]
    
# ==================================
# Convert Date Column
# ==================================
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True,
    errors="coerce"
)

# Remove invalid dates if any
df = df.dropna(subset=["Order Date"])

# ==================================
# Dashboard Title
# ==================================
st.title("📊 Sales Analytics Dashboard")

# ==================================
# KPI Cards
# ==================================
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Total Orders", total_orders)


# ==================================
# Monthly Sales Trend
# ==================================
df["Month_Num"] = df["Order Date"].dt.month
df["Month"] = df["Order Date"].dt.month_name()

monthly_sales = (
    df.groupby(["Month_Num", "Month"])["Sales"]
    .sum()
    .reset_index()
    .sort_values("Month_Num")
)

fig1 = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    title="Monthly Sales Trend"
)

st.plotly_chart(fig1, use_container_width=True)

# ==================================
# Top 10 Products by Sales
# ==================================
top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig2 = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    title="Top 10 Products by Sales"
)

st.plotly_chart(fig2, use_container_width=True)

# ==================================
# Sales by Customer Segment
# ==================================
segment_sales = (
    df.groupby("Segment")["Sales"]
    .sum()
    .reset_index()
)

fig3 = px.pie(
    segment_sales,
    names="Segment",
    values="Sales",
    title="Sales by Customer Segment"
)

st.plotly_chart(fig3, use_container_width=True)

# ==================================
# Top 10 States by Sales
# ==================================
state_sales = (
    df.groupby("State/Province")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig4 = px.bar(
    state_sales,
    x="State/Province",
    y="Sales",
    title="Top 10 States by Sales"
)

st.plotly_chart(fig4, use_container_width=True)


profit_category = (
    df.groupby("Category")["Profit"]
      .sum()
      .reset_index()
)

fig5 = px.bar(
    profit_category,
    x="Category",
    y="Profit",
    title="Profit by Category"
)

st.plotly_chart(fig5, use_container_width=True)
# --------------------------------
# Top 10 Customers by Sales
# --------------------------------
st.subheader("Top 10 Customers")

top_customers = (
    df.groupby("Customer Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

fig = px.bar(
    top_customers,
    x="Sales",
    y="Customer Name",
    orientation="h",
    title="Top Customers by Sales"
)
# ==================================
# SALES FORECAST
# ==================================

st.subheader("Sales Forecast")

forecast_data = (
    full_df.groupby("Order Date")["Sales"]
           .sum()
           .reset_index()
)
forecast_data["Order Date"] = pd.to_datetime(
    forecast_data["Order Date"],
    dayfirst=True
)
forecast_data.columns = ["ds", "y"]

if len(forecast_data) > 2:

    model = Prophet()

    model.fit(forecast_data)

    future = model.make_future_dataframe(
        periods=30
    )

    forecast = model.predict(future)

    fig_forecast = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="Next 30 Days Sales Forecast"
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

else:
    st.warning(
        "Not enough data available for forecasting. Try removing some filters."
    )
st.plotly_chart(fig, use_container_width=True)

st.download_button(
    label="Download Data",
    data=df.to_csv(index=False),
    file_name="sales_report.csv",
    mime="text/csv"
)
# Dataset Preview
st.subheader("Dataset Preview")

with st.expander("📂 View Dataset"):
    st.dataframe(df)
# footer
st.markdown("---")
st.markdown(
    "Created by Karan | Streamlit • Pandas • Plotly"
)