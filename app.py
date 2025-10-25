from io import BytesIO
from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
from fpdf import FPDF
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load your model
loaded_data = joblib.load('stroke_model.pkl')
model = loaded_data['model']  # Adjust key if needed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    form_data = request.form

    input_data = [
        form_data['gender'],
        int(form_data['age']),
        int(form_data['hypertension']),
        int(form_data['heart_disease']),
        form_data['ever_married'],
        form_data['work_type'],
        form_data['residence_type'],
        float(form_data['avg_glucose_level']),
        float(form_data['bmi']),
        form_data['smoking_status']
    ]

    # Encoding
    mapping = {
        'Male': 1, 'Female': 0,
        'Yes': 1, 'No': 0,
        'Urban': 1, 'Rural': 0,
        'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3,
        'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3
    }

    encoded_data = [
        mapping.get(input_data[0], 0),
        input_data[1],
        input_data[2],
        input_data[3],
        mapping.get(input_data[4], 0),
        mapping.get(input_data[5], 0),
        mapping.get(input_data[6], 0),
        input_data[7],
        input_data[8],
        mapping.get(input_data[9], 0)
    ]

    features = np.array(encoded_data).reshape(1, -1)
    result = model.predict(features)[0]

    if result < 0.5:
        pred = "No Stroke"
        diet_plan = """<h4>Diet Plan - For Low-Risk / No Stroke Patients:</h4>
        <ul>
            <li><strong>Balanced Diet:</strong> Eat more fruits, vegetables, whole grains, and less oily or salty foods.</li>
            <li><strong>Regular Exercise:</strong> At least 30 mins/day of walking, yoga, or cardio.</li>
            <li>zAvoid Risky Habits:</strong> No smoking, limit alcohol.</li>
            <li>Manage Stress:</strong> Practice meditation, breathing exercises.</li>
            <li>Routine Checkups:</strong> Monitor blood pressure, cholesterol, and BMI regularly.</li>
        </ul>"""
    else:
        pred = "Stroke"
        diet_plan = """<h4 style="color: red;">Diet Plan - For High-Risk / Stroke Patients:</h4>
        <ul>

        <ul>
            <li><strong>Medication Adherence: </strong>Take your medicines every day without missing.</li>
            <li><strong>Low Sodium Diet:</strong> Strict salt reduction and fluid intake control.</li>
            <li><strong>Supervised Physical Therapy:</strong> Gentle, guided recovery exercises.</li>
            <li><strong>Cognitive Health Support:</strong> Do brain games or memory exercises to keep your mind sharp.</li>
            <li><strong>Frequent Monitoring:</strong> Regular visits to neurologist/cardiologist.</li>
            <li><strong>Family Education:</strong> Involve caregivers for emergency response training and support.</li>
        </ul>"""

    return render_template(
        'index.html',
        pred=pred,
        prediction=diet_plan,
        gender=input_data[0],
        age=input_data[1],
        hypertension=input_data[2],
        heart_disease=input_data[3],
        ever_married=input_data[4],
        work_type=input_data[5],
        residence_type=input_data[6],
        avg_glucose_level=input_data[7],
        bmi=input_data[8],
        smoking_status=input_data[9]
    )

# PDF Generator Class
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", size=16)
        self.cell(0, 10, "Brain Stroke Prediction Report", ln=True, align='C')

    def add_text(self, title, value):
        self.set_font("Arial", size=12)
        self.cell(0, 10, f"{title}: {value}", ln=True)

    def add_diet_text(self, html_text):
        self.set_font("Arial", size=12)
        soup = BeautifulSoup(html_text, 'html.parser')
        for li in soup.find_all('li'):
            line = li.get_text()
            line = line.encode('latin-1', 'replace').decode('latin-1')  # replace unsupported characters
            self.multi_cell(0, 10, f"- {line}")

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    pdf = PDF()
    pdf.add_page()

    # Add input values
    input_fields = [
        ("Gender", request.form['gender']),
        ("Age", request.form['age']),
        ("Hypertension", request.form['hypertension']),
        ("Heart Disease", request.form['heart_disease']),
        ("Ever Married", request.form['ever_married']),
        ("Work Type", request.form['work_type']),
        ("Residence Type", request.form['residence_type']),
        ("Avg Glucose Level", request.form['avg_glucose_level']),
        ("BMI", request.form['bmi']),
        ("Smoking Status", request.form['smoking_status']),
        ("Prediction Result", request.form['result'])
    ]

    for title, value in input_fields:
        pdf.add_text(title, value)

    # Add diet plan
    pdf.ln(10)
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Diet Plan", ln=True)
    pdf.ln(5)
    pdf.add_diet_text(request.form['diet_plan'])

    # Create PDF in memory (use 'S' to store in string form)
    pdf_output = pdf.output(dest='S').encode('latin-1')

    # Return the PDF file
    return send_file(BytesIO(pdf_output), as_attachment=True, download_name="stroke_prediction_report.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
