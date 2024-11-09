# create_database.py

import sqlite3

import pandas as pd

# Step 1: Read the CSV file
csv_file_path = r"B:\Documents\MINI PROJECT REPORTS\SEM 7 MINI PROJECT\BD\World-happiness-report-2024.csv"  # Update this to your CSV file path
data = pd.read_csv(csv_file_path)

# Step 2: Rename columns to remove spaces
data.columns = data.columns.str.replace(' ', '_')  # Replace spaces with underscores

# Step 3: Connect to SQLite database
conn = sqlite3.connect('world_happiness_report.db')  # This creates a new database file

# Step 4: Write the DataFrame to a new SQL table
data.to_sql('HappinessData12', conn, if_exists='replace', index=False)

# Optional: Check if the table has been created successfully
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(query, conn)
print("Tables in the database:", tables)

# Step 5: Close the database connection
conn.close()
