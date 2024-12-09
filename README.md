# Flight Price Prediction Model

## Objective
Predict flight prices based on journey and flight details to help users estimate costs and make better travel decisions.

## Dataset Overview
Flight Details:
Airline, Route, Source, Destination, Total_Stops.
Journey Information:
Date_of_Journey, Dep_Time, Arrival_Time, Duration.
Additional Features:
Additional_Info (e.g., meal availability, in-flight services).
Target Variable:
Price (flight ticket cost in currency).

## Problem Statements
1. Perform Feature Engineering
2. Identify the most preferred airline.
3. Find the most frequent departure sources and arrival destinations.
4. Analyze the impact of features like `Total_Stops`, `Duration`, and `Airline` on flight prices.
5. Prepare the data with feature encoding for machine learning models.

## Workflow
1. **Data Preprocessing**: Handle missing values, extract date/time features, encode categorical variables.
2. **EDA**: Visualize trends and relationships between features and `Price`.
3. **Model Training**: Train regression models like Linear Regression, Random Forest, and XGBoost.
4. **Evaluation**: Assess model performance using MAE and RMSE.


