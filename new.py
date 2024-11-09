import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Connect to SQLite database
conn = sqlite3.connect('world_happiness_report.db')

# Function to execute SQL query and return DataFrame
def run_query(query):
    return pd.read_sql_query(query, conn)

# Function to calculate and display model metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"B:\Documents\MINI PROJECT REPORTS\SEM 7 MINI PROJECT\BD\World-happiness-report-2024.csv")
    return df

# Streamlit app title
st.title("World Happiness Report Analysis")

# Sidebar for model selection
st.sidebar.title("Regression Model Comparison")
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
}

# Sidebar for selecting models to compare
selected_models = st.sidebar.multiselect("Select Models", list(models.keys()), default=["Linear Regression"])

# Load data
data = load_data()
if not data.empty:
    st.write("Dataset Loaded Successfully")

    # Display dataset if required
    if st.checkbox('Show Dataset'):
        st.write(data)

    # Feature and target variable selection
    feature_columns = st.multiselect("Select Features", data.columns)
    target_column = st.selectbox("Select Target Variable", data.columns)

    if feature_columns and target_column:
        X = data[feature_columns]
        y = data[target_column]

        # Handle missing values by filling with the mean directly
        X = X.fillna(X.mean())

        # Compute means of the features (after X is defined)
        X_means = X.mean()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize metrics dataframe
        metrics_df = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R²"])

        # Iterate over selected models and evaluate
        for model_name in selected_models:
            model = models[model_name]
            mae, mse, rmse, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
            new_row = pd.DataFrame({
                "Model": [model_name],
                "MAE": [mae],
                "MSE": [mse],
                "RMSE": [rmse],
                "R²": [r2]
            })
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        # Display results
        st.write("Model Performance:")
        st.dataframe(metrics_df)

        # Plot results
        st.bar_chart(metrics_df.set_index("Model")[["MAE", "MSE", "RMSE"]])

        # New option for comparison plot
        if st.sidebar.button("View Comparison Plot"):
            st.subheader("Comparison of Model Performance Metrics")
            
            # Create subplots
            fig, axs = plt.subplots(4, 1, figsize=(10, 12))
            axs[0].barh(metrics_df["Model"], metrics_df["MAE"], color='orange')
            axs[0].set_title('Mean Absolute Error (MAE)')
            axs[0].set_xlim(0, metrics_df["MAE"].max() + 0.1)

            axs[1].barh(metrics_df["Model"], metrics_df["RMSE"], color='blue')
            axs[1].set_title('Root Mean Squared Error (RMSE)')
            axs[1].set_xlim(0, metrics_df["RMSE"].max() + 0.1)

            axs[2].barh(metrics_df["Model"], metrics_df["R²"], color='green')
            axs[2].set_title('R² Score')
            axs[2].set_xlim(0, 1)

            axs[3].barh(metrics_df["Model"], metrics_df["MSE"], color='red')
            axs[3].set_title('Mean Squared Error (MSE)')
            axs[3].set_xlim(0, metrics_df["MSE"].max() + 0.1)

            for ax in axs:
                ax.set_xlabel('Score')
                ax.set_ylabel('Models')

            plt.tight_layout()
            st.pyplot(fig)  # Display the plot

        # Add input for user-defined feature values
        st.subheader("Enter Feature Values for Prediction")
        input_data = {}

        # Create input fields for each feature
        for feature in feature_columns:
            input_data[feature] = st.number_input(f"Enter value for {feature}:", value=float(X_means[feature]), step=0.01)

        # Prepare the input DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict using the first selected model if available
        if selected_models:
            model = models[selected_models[0]]
            predicted_score = model.predict(input_df)

            # Determine the happiness category based on the predicted score
            if predicted_score[0] >= 6.0:  # Adjust this threshold as needed
                happiness_status = "😀"  # High happiness
                emoji_size = "<h1 style='font-size:50px;'>😀</h1>"  # Big emoji
                status_text = "High Happiness"
            elif predicted_score[0] >= 4.0:  # Adjust this threshold as needed
                happiness_status = "🙂"  # Average happiness
                emoji_size = "<h1 style='font-size:50px;'>🙂</h1>"  # Big emoji
                status_text = "Average Happiness"
            else:
                happiness_status = "🥲"  # Low happiness
                emoji_size = "<h1 style='font-size:50px;'>🥲</h1>"  # Big emoji
                status_text = "Low Happiness"

            # Display the predicted happiness score and status
            st.write(f"Predicted Happiness Score: {predicted_score[0]:.2f}")
            st.markdown(emoji_size, unsafe_allow_html=True)  # Display the emoji
            st.write(status_text)  # Display the corresponding status text

else:
    st.write("No data available. Please load the dataset.")

# Placeholder for metrics
st.subheader("Model Performance Metrics")

# Queries
queries = {
    "Happiest Country": "SELECT Country_name, Ladder_score FROM HappinessData12 ORDER BY Ladder_score DESC LIMIT 1;",
    "Average Happiest Country": "SELECT AVG(Ladder_score) AS AverageHappiness FROM HappinessData12;",
    "Least Happiest Country": "SELECT Country_name, Ladder_score FROM HappinessData12 ORDER BY Ladder_score ASC LIMIT 1;",
    "Top 10 Highest GDP Countries": "SELECT Country_name, Log_GDP_per_capita FROM HappinessData12 ORDER BY Log_GDP_per_capita DESC LIMIT 10;",
    "Top 10 Lowest GDP Countries": "SELECT Country_name, Log_GDP_per_capita FROM HappinessData12 ORDER BY Log_GDP_per_capita ASC LIMIT 10;",
    "Average Happiness Score by Region": "SELECT Regional_indicator, AVG(Ladder_score) AS AverageHappiness FROM HappinessData12 GROUP BY Regional_indicator ORDER BY AverageHappiness DESC;",
    "Top 10 Countries with Best Social Support": "SELECT Country_name, Social_support FROM HappinessData12 ORDER BY Social_support DESC LIMIT 10;",
    "Top 10 Countries with Highest Freedom": "SELECT Country_name, Freedom_to_make_life_choices FROM HappinessData12 ORDER BY Freedom_to_make_life_choices DESC LIMIT 10;",
    "Top 10 Countries with Lowest Corruption": "SELECT Country_name, Perceptions_of_corruption FROM HappinessData12 ORDER BY Perceptions_of_corruption ASC LIMIT 10;",
    "Top 5 Countries for Generosity": "SELECT Country_name, Generosity FROM HappinessData12 ORDER BY Generosity DESC LIMIT 5;",
    "Countries Above Average Happiness": "SELECT Country_name, Ladder_score FROM HappinessData12 WHERE Ladder_score > (SELECT AVG(Ladder_score) FROM HappinessData12);"
}

# Create sidebar for queries
st.sidebar.header("Select a Query")
query_selection = st.sidebar.selectbox("Choose a query to display results:", list(queries.keys()))

# Execute the selected query and display results
if query_selection:
    sql_query = queries[query_selection]
    result_df = run_query(sql_query)
    
    # Display the result in the Streamlit app
    st.subheader(query_selection)
    st.dataframe(result_df)
