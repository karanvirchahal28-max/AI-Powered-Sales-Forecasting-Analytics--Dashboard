import pandas as pd

df = pd.read_csv("data/sales_data.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
df.dropna(inplace=True)
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Year"] = df["Order Date"].dt.year

df["Month"] = df["Order Date"].dt.month_name()
print(df[["Order Date", "Year", "Month"]].head())
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique()

print("Total Sales =", total_sales)
print("Total Profit =", total_profit)
print("Total Orders =", total_orders)

df["Month_Num"] = df["Order Date"].dt.month

monthly_sales = (
    df.groupby(["Month_Num", "Month"])["Sales"]
      .sum()
      .reset_index()
      .sort_values("Month_Num")
)

print(monthly_sales)
import plotly.express as px

fig = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    title="Monthly Sales Trend"
)

fig.show() 
# Step 1: Top 10 Products by Sales
top_products = (
    df.groupby("Product Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

print(top_products) 
# Step 2: Product Sales Chart

fig = px.bar(
    top_products,
    x="Product Name",
    y="Sales",
    title="Top 10 Products by Sales"
)

fig.show()

# Step 3: Category Analysis

category_sales = (
    df.groupby("Category")["Sales"]
      .sum()
      .reset_index()
)

print(category_sales)

# Step 4: Category Pie Chart


fig = px.pie(
    category_sales,
    names="Category",
    values="Sales",
    title="Sales by Category"
)

fig.show()
#Step 5: State-wise Sales
state_sales = (
    df.groupby("State/Province")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

print(state_sales)

#Step 6: State-wise Chart
fig = px.bar(
    state_sales,
    x="State/Province",
    y="Sales",
    title="Top 10 States by Sales"
)

fig.show()

#Step 7: Customer Segment Analysis
segment_sales = (
    df.groupby("Segment")["Sales"]
      .sum()
      .reset_index()
)

print(segment_sales)

fig = px.pie(
    segment_sales,
    names="Segment",
    values="Sales",
    title="Sales by Customer Segment"
)

fig.show()

