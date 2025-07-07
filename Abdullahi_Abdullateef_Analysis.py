import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


#Setting up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

#Loading the datasets
print("Loading datasets...")
df_sales = pd.read_excel('Ace Superstore Retail Dataset.xlsx')
df_locations = pd.read_excel('Store Locations.xlsx')


#Displaying basic information about the datasets
print("\n  DATASET OVERVIEW ")
print(f"Sales Dataset Shape: {df_sales.shape}")
print(f"Locations Dataset Shape: {df_locations.shape}")
print(f"\nSales Dataset Columns: {list(df_sales.columns)}")
print(f"\nFirst few rows of sales data:")
print(df_sales.head())

# Data preprocessing
print("\n DATA PREPROCESSING ")

# Converting Order Date to datetime
df_sales['Order Date'] = pd.to_datetime(df_sales['Order Date'])

# Checking for missing values
print("\nMissing Values:")
print(df_sales.isnull().sum())

# Basic data quality checks
print(f"\nUnique Order IDs: {df_sales['Order ID'].nunique()}")
print(f"Total Records: {len(df_sales)}")
print(f"Date Range: {df_sales['Order Date'].min()} to {df_sales['Order Date'].max()}")

# Calculating profit margin
df_sales['Profit'] = df_sales['Sales'] - df_sales['Cost Price']
df_sales['Profit_Margin'] = (df_sales['Profit'] / df_sales['Sales']) * 100

# Adding the year and month columns for time analysis
df_sales['Year'] = df_sales['Order Date'].dt.year
df_sales['Month'] = df_sales['Order Date'].dt.month
df_sales['Month_Year'] = df_sales['Order Date'].dt.to_period('M')

print("\n KEY BUSINESS METRICS ")

# 1. Total Sales, Revenue, and Discount Rates by Region and Segment
print("\n1. SALES PERFORMANCE BY REGION")
region_summary = df_sales.groupby('Region').agg({
    'Sales': ['sum', 'count', 'mean'],
    'Cost Price': 'sum',
    'Profit': 'sum',
    'Discount': 'mean',
    'Quantity': 'sum'
}).round(2)

region_summary.columns = ['Total_Sales', 'Total_Orders', 'Avg_Order_Value', 
                         'Total_Cost', 'Total_Profit', 'Avg_Discount_Rate', 'Total_Quantity']

print(region_summary.sort_values('Total_Sales', ascending=False))

# 2. Sales Performance by Order Mode (Online vs In-Store)
print("\n2. SALES PERFORMANCE BY ORDER MODE")
order_mode_summary = df_sales.groupby('Order Mode').agg({
    'Sales': ['sum', 'count', 'mean'],
    'Profit': 'sum',
    'Discount': 'mean',
    'Quantity': 'sum'
}).round(2)

order_mode_summary.columns = ['Total_Sales', 'Total_Orders', 'Avg_Order_Value', 
                             'Total_Profit', 'Avg_Discount_Rate', 'Total_Quantity']

print(order_mode_summary)

# 3. Top 5 Best-Selling Products by Revenue
print("\n3. TOP 5 BEST-SELLING PRODUCTS BY REVENUE")
top_products = df_sales.groupby(['Product ID', 'Product Name']).agg({
    'Sales': 'sum',
    'Quantity': 'sum',
    'Profit': 'sum'
}).sort_values('Sales', ascending=False).head(5)

print(top_products)

# 4. Bottom 5 Underperforming Products by Revenue
print("\n4. BOTTOM 5 UNDERPERFORMING PRODUCTS BY REVENUE")
bottom_products = df_sales.groupby(['Product ID', 'Product Name']).agg({
    'Sales': 'sum',
    'Quantity': 'sum',
    'Profit': 'sum'
}).sort_values('Sales', ascending=True).head(5)

print(bottom_products)

# 5. Product Categories with Highest Margins
print("\n5. PRODUCT CATEGORIES WITH HIGHEST PROFIT MARGINS")
category_margins = df_sales.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Profit_Margin': 'mean',
    'Quantity': 'sum'
}).round(2)

category_margins['Profit_Margin_Calc'] = (category_margins['Profit'] / category_margins['Sales'] * 100).round(2)
category_margins = category_margins.sort_values('Profit_Margin_Calc', ascending=False)

print(category_margins)

# Creating  visualizations
print("\n CREATING VISUALIZATIONS ")

# Visualization 1: Sales by Region
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
region_sales = df_sales.groupby('Region')['Sales'].sum().sort_values(ascending=False)
region_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Sales (£)')
plt.xticks(rotation=45)

# Visualization 2: Sales by Order Mode
plt.subplot(2, 2, 2)
order_mode_sales = df_sales.groupby('Order Mode')['Sales'].sum()
plt.pie(order_mode_sales.values, labels=order_mode_sales.index, autopct='%1.1f%%', startangle=90)
plt.title('Sales Distribution by Order Mode')

# Visualization 3: Top 10 Categories by Sales
plt.subplot(2, 2, 3)
top_categories = df_sales.groupby('Category')['Sales'].sum().sort_values(ascending=False).head(10)
top_categories.plot(kind='barh', color='lightgreen')
plt.title('Top 10 Categories by Sales')
plt.xlabel('Sales (£)')

# Visualization 4: Monthly Sales Trend
plt.subplot(2, 2, 4)
monthly_sales = df_sales.groupby('Month_Year')['Sales'].sum()
monthly_sales.plot(kind='line', marker='o', color='coral')
plt.title('Monthly Sales Trend')
plt.xlabel('Month-Year')
plt.ylabel('Sales (£)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Additional Analysis: Discount Impact
print("\n DISCOUNT ANALYSIS ")
discount_bins = pd.cut(df_sales['Discount'], bins=[0, 0.1, 0.2, 0.3, 1.0], 
                      labels=['0-10%', '10-20%', '20-30%', '30%+'])
discount_impact = df_sales.groupby(discount_bins).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).round(2)

print("Impact of Discount Rates:")
print(discount_impact)

# Geographic Analysis
print("\n GEOGRAPHIC ANALYSIS ")
city_performance = df_sales.groupby('City').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).sort_values('Sales', ascending=False).head(10)

print("Top 10 Cities by Sales:")
print(city_performance)

# Summary Statistics
print("\n EXECUTIVE SUMMARY ")
print(f"Total Revenue: £{df_sales['Sales'].sum():,.2f}")
print(f"Total Profit: £{df_sales['Profit'].sum():,.2f}")
print(f"Overall Profit Margin: {(df_sales['Profit'].sum() / df_sales['Sales'].sum() * 100):.2f}%")
print(f"Total Orders: {len(df_sales):,}")
print(f"Average Order Value: £{df_sales['Sales'].mean():.2f}")
print(f"Average Discount Rate: {df_sales['Discount'].mean():.2%}")
print(f"Total Products Sold: {df_sales['Quantity'].sum():,}")

#Save results to CSV for further analysis
region_summary.to_csv('region_analysis.csv')
category_margins.to_csv('category_margins.csv')
top_products.to_csv('top_products.csv')

print("\n ANALYSIS COMPLETE ")
print("Results saved to CSV files for further reference.")
print("GitHub repository: RDAMP-Sales-Analysis")
