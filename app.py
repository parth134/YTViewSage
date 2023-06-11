import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to preprocess the data
def preprocess_data(df):
    # Convert 'views', 'likes', 'dislikes', 'comment', and 'duration' to numeric
    df['views'] = pd.to_numeric(df['views'], errors='coerce')
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
    df['dislikes'] = pd.to_numeric(df['dislikes'], errors='coerce')
    df['comment'] = pd.to_numeric(df['comment'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    # Convert 'published' to datetime
    df['published'] = pd.to_datetime(df['published'])

    # Convert 'category' to categorical
    df['category'] = df['category'].astype('category')

    return df

# Function to train the model and make predictions
def train_and_predict(df):
    # Split the data into features and target
    X = df[['views', 'likes', 'dislikes', 'comment', 'duration']]
    y = df['adview']

    # Create an imputer to fill in missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, y_pred, mse, r2, y_test

# Main function
def main():
    st.title("YouTube Adview Prediction")
    st.subheader("Dataset Exploration")

    # Load the dataset
    try:
        df = pd.read_csv("train-adview.csv")
    except FileNotFoundError:
        st.error("Dataset file not found.")
        return
    except pd.errors.EmptyDataError:
        st.error("Dataset file is empty.")
        return
    except pd.errors.ParserError:
        st.error("Unable to read the dataset file.")
        return

    # Display the dataset
    st.write(df)

    # Preprocess the data
    df = preprocess_data(df)

    st.subheader("Adview Prediction")

    # Train the model and make predictions
    try:
        model, y_pred, mse, r2, y_test = train_and_predict(df)
    except ValueError as e:
        st.error(f"Error occurred during model training: {str(e)}")
        return

    # Display the results
    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r2)

    # Plot the predicted vs actual adviews
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual Adviews')
    ax.set_ylabel('Predicted Adviews')
    ax.set_title('Actual vs Predicted Adviews')
    st.pyplot(fig)

    # Save the figure
    plt.savefig("adviews_prediction_plot.png")

    st.subheader("Adview Prediction for New Video")

    # Get user input for new video features
    views = st.number_input("Enter number of views:")
    likes = st.number_input("Enter number of likes:")
    dislikes = st.number_input("Enter number of dislikes:")
    comments = st.number_input("Enter number of comments:")
    duration = st.number_input("Enter video duration (in seconds):")

    # Preprocess the user input
    input_data = pd.DataFrame({
        'views': [views],
        'likes': [likes],
        'dislikes': [dislikes],
        'comment': [comments],
        'duration': [duration]
    })

    input_data = preprocess_data(input_data)

    # Make prediction for the new video
    try:
        prediction = model.predict(input_data)
        st.subheader("Adview Prediction Result")
        st.write("Predicted Adviews:", prediction[0])
    except ValueError as e:
        st.error(f"Error occurred during prediction: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
