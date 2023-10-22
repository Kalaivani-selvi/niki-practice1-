from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the dataset (for demonstration purposes)
data = pd.read_csv("Salary Data.csv")  # Replace with the correct dataset path

# Preprocess the data
imputer = SimpleImputer(strategy='mean')
data[['Age', 'Years of Experience']] = imputer.fit_transform(data[['Age', 'Years of Experience']])

# Impute missing salary values
imputer_salary = SimpleImputer(strategy='mean')
data['Salary'] = imputer_salary.fit_transform(data[['Salary']])

# Encode categorical features
label_encoder_job = LabelEncoder()
data['Job Encoded'] = label_encoder_job.fit_transform(data['Job Title'])

label_encoder_education = LabelEncoder()
data['Education Encoded'] = label_encoder_education.fit_transform(data['Education Level'])

X = np.column_stack((data['Age'], data['Years of Experience'], data['Job Encoded'], data['Education Encoded']))
y = data['Salary']

# Get unique job titles from the dataset
unique_job_titles = data['Job Title'].unique()

unique_education = data['Education Level'].unique()

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_salary = None

    if request.method == "POST":
        age = float(request.form["age"])
        years_of_experience = float(request.form["experience"])
        job = request.form["job"]
        education = request.form["education"]

        # Handle unseen labels using try-except
        try:
            job_encoded = label_encoder_job.transform([job])[0]
            education_encoded = label_encoder_education.transform([education])[0]
            input_data = np.array([[age, years_of_experience, job_encoded, education_encoded]])
            predicted_salary = model.predict(input_data)[0]
        except ValueError:
            # Handle the case of an unseen label
            predicted_salary = "Label not recognized"

    return render_template("index.html", predicted_salary=predicted_salary, unique_job_titles=unique_job_titles,unique_education=unique_education)

if __name__ == "__main__":
    app.run(debug=True)
