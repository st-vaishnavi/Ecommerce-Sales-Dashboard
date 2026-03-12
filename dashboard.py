import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np  
st.set_page_config(page_title="E-commerce Sales Dashboard", layout="wide")

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
df = pd.read_csv("cleaned_sales_data.csv")

# Convert date column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ------------------------------
# Step 2: Dashboard Title
# ------------------------------

st.title("📊 E-Commerce Sales Dashboard")
st.write("Interactive dashboard for Sales Data Analysis")

# ------------------------------
# Step 3: Sidebar Filters
# ------------------------------
st.sidebar.header("Filter Options")

# Country Filter (instead of Region)
countries = df['Country'].unique().tolist()
selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries)

# Product Filter (Description)
products = df['Description'].unique().tolist()
selected_products = st.sidebar.multiselect(
    "Select Products", products, default=products[:10]
)

# Date Filter
start_date = st.sidebar.date_input("Start Date", df['InvoiceDate'].min())
end_date = st.sidebar.date_input("End Date", df['InvoiceDate'].max())

# Apply filters
filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

filtered_df = filtered_df[filtered_df['Description'].isin(selected_products)]

filtered_df = filtered_df[
    (filtered_df['InvoiceDate'] >= pd.to_datetime(start_date)) &
    (filtered_df['InvoiceDate'] <= pd.to_datetime(end_date))
]

# ------------------------------
# Step 4: Key Metrics (KPIs)
# ------------------------------
total_sales = filtered_df['TotalPrice'].sum()
total_orders = filtered_df['InvoiceNo'].nunique()
average_order_value = filtered_df['TotalPrice'].mean()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
col2.metric("🛒 Total Orders", total_orders)
col3.metric("📈 Avg Order Value", f"${average_order_value:,.2f}")

# ------------------------------
# Step 5: Tabs for Charts
# ------------------------------
tab1, tab2 = st.tabs(["Sales Overview", "Product Analysis"])

# --- Tab 1: Sales Overview ---
with tab1:
    col1, col2 = st.columns(2)

    # --- Chart 1: Sales Over Time ---
    sales_time = filtered_df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()

    fig1 = px.line(
        sales_time,
        x='InvoiceDate',
        y='TotalPrice',
        title="Sales Over Time",
        markers=True,
        template="plotly_white"
    )

    col1.plotly_chart(fig1, use_container_width=True)

    # --- Chart 2: Sales by Country ---
    sales_country = filtered_df.groupby('Country')['TotalPrice'].sum().reset_index()

    fig2 = px.bar(
        sales_country,
        x='Country',
        y='TotalPrice',
        title="Sales by Country",
        template="plotly_white",
        color='TotalPrice'
    )

    col2.plotly_chart(fig2, use_container_width=True)
    st.subheader("Monthly Sales Trend")

monthly_sales = filtered_df.copy()
monthly_sales["Month"] = monthly_sales["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = monthly_sales.groupby("Month")["TotalPrice"].sum().reset_index()

fig_month = px.line(
    monthly_sales,
    x="Month",
    y="TotalPrice",
    title="Monthly Sales Trend",
    markers=True,
    template="plotly_white"
)

st.plotly_chart(fig_month, use_container_width=True)
st.subheader("Future Sales Prediction")

# Prepare monthly data
prediction_df = filtered_df.copy()
prediction_df["Month"] = prediction_df["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = prediction_df.groupby("Month")["TotalPrice"].sum().reset_index()

# Create index for ML
monthly_sales["MonthIndex"] = np.arange(len(monthly_sales))

X = monthly_sales[["MonthIndex"]]
y = monthly_sales["TotalPrice"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 6 months
future_index = np.arange(len(monthly_sales), len(monthly_sales)+6).reshape(-1,1)
predictions = model.predict(future_index)

# Create future dataframe
future_months = pd.DataFrame({
    "Month": [f"Future {i}" for i in range(1,7)],
    "TotalPrice": predictions
})

# Combine past + future
combined = pd.concat([monthly_sales[["Month","TotalPrice"]], future_months])

# Plot prediction
fig_pred = px.line(
    combined,
    x="Month",
    y="TotalPrice",
    title="Sales Forecast (Next 6 Months)",
    markers=True,
    template="plotly_white"
)

st.plotly_chart(fig_pred, use_container_width=True)

    
# --- Tab 2: Product Analysis ---
with tab2:
    
    col1, col2 = st.columns(2)

    # Top Products by Sales
    sales_product = filtered_df.groupby('Description')['TotalPrice'].sum()\
        .sort_values(ascending=False).head(10).reset_index()

    fig3 = px.bar(
        sales_product,
        x='Description',
        y='TotalPrice',
        title="Top Products by Sales",
        template="plotly_white",
        color='TotalPrice'
    )

    col1.plotly_chart(fig3, use_container_width=True)

    # Top Products by Quantity
    quantity_product = filtered_df.groupby('Description')['Quantity'].sum()\
        .sort_values(ascending=False).head(10).reset_index()

    fig4 = px.bar(
        quantity_product,
        x='Description',
        y='Quantity',
        title="Top Products by Quantity",
        template="plotly_white",
        color='Quantity'
    )

    col2.plotly_chart(fig4, use_container_width=True)
    # Top 10 Selling Products (Horizontal Chart)
    st.subheader("Top 10 Selling Products")
top_products = filtered_df.groupby("Description")["Quantity"]\
    .sum().sort_values(ascending=False).head(10).reset_index()

fig5 = px.bar(
    top_products,
    x="Quantity",
    y="Description",
    orientation="h",
    title="Top 10 Selling Products",
    template="plotly_white",
    color="Quantity"
)

st.plotly_chart(fig5, use_container_width=True)


# ------------------------------
# Step 6: Show Data Table
# ------------------------------
st.subheader("Filtered Data")

st.dataframe(filtered_df)

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_sales.csv",
    mime="text/csv"
)
