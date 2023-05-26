import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route('/')
def placement_prediction():
    return render_template('/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the form inputs from the request
    ssc_p = float(request.form['ssc_p'])
    hsc_p = float(request.form['hsc_p'])
    degree_p = float(request.form['degree_p'])
    workex = int(request.form['workex'])
    test_score = float(request.form['test_score'])

    df = pd.read_csv("Placement_Data_Full_Class.csv")
    df.drop(
        columns=['ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'specialisation', 'mba_p', 'sl_no', 'gender', 'salary'],
        inplace=True)

    df['status'].replace('Placed', 1, inplace=True)
    df['status'].replace('Not Placed', 0, inplace=True)
    df['workex'].replace('No', 0, inplace=True)
    df['workex'].replace('Yes', 1, inplace=True)
    df.rename(columns={'etest_p': 'test_score'}, inplace=True)

    # Save the modified DataFrame back to the same CSV file
    df.to_csv('your_file.csv', index=False)

    # Split the data into features (X) and labels (y)
    X = df[['ssc_p', 'hsc_p', 'degree_p', 'workex', 'test_score']]
    y = df['status']

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initialize logistic regression classifier and fit the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Prepare the new input values as a DataFrame
    new_input_df = pd.DataFrame({'ssc_p': [ssc_p],
                                 'hsc_p': [hsc_p],
                                 'degree_p': [degree_p],
                                 'workex': [workex],
                                 'test_score': [test_score]})

    # Make prediction using the trained model
    prediction = model.predict(new_input_df)[0]

    # Set placement message based on the prediction
    placement_message = ''
    if prediction == 1:
        placement_message = 'Your chances are high to get placed!'
    else:
        placement_message = 'Your chances of placement are low.\n'

        # Get the lowest value and its column name
        if placement_message == 'Your chances of placement are low.\n':
            lowest_value = new_input_df.min().min()
            lowest_value_column = new_input_df.min().idxmin()

            # Print the lowest value and its column name
            print("Areas to Focus: \n")
            if lowest_value_column == 'ssc_p' or lowest_value_column == 'hsc_p':
                m = "\n 1) Add Projects in your Resume\n 2) Try To Increase Your Test Score\n"
                placement_message = placement_message + m
            elif lowest_value_column == 'workex':
                m = "\n Try Extra Curricular and More Work Experience in Your Resume"
                placement_message = placement_message + m
            else:
                m = "\n Try to increase your existing grades and test score"
                placement_message = placement_message + m


    

    return render_template('index.html', prediction=prediction, placement_message=placement_message)


if __name__ == '__main__':
    app.run(debug=True)
