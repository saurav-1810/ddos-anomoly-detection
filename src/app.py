import os
import joblib
import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_from_directory
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor

matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained models
svm_model = joblib.load(os.path.join(BASE_DIR, 'model', 'svm_model.pkl'))
rf_model = joblib.load(os.path.join(BASE_DIR, 'model', 'random_forest_model.pkl'))

# Define the feature columns
features = [' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
            ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
            ' Fwd Packet Length Max', ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
            'Bwd Packet Length Max', ' Bwd Packet Length Min',
            ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
            'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
            ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
            ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
            ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
            ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
            ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
            ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
            ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
            ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
            ' ECE Flag Count']

# Number of threads to use for SVM predictions
NUM_THREADS = 10  # Adjust this number as needed


def predict_svm_in_chunks(X_scaled):
    chunks = np.array_split(X_scaled, 20)  # Adjust chunk size as needed
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(svm_model.predict, chunks))
    return np.concatenate(results)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        df = pd.read_csv(file)

        # Extract the feature columns
        X = df[features]

        # Standardize the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Make predictions
        rf_pred = rf_model.predict(X_scaled)
        svm_pred = predict_svm_in_chunks(X_scaled)

        # Final prediction based on SVM and RF agreement
        combined_pred = np.where((rf_pred == 1) | (svm_pred == 1), 'DDoS', 'BENIGN')

        # Save predictions to CSV
        df['Prediction'] = combined_pred  # Ensure predictions are assigned correctly
        df_ddos = df[df['Prediction'] == 'DDoS']  # Correctly filter DDoS rows
        df_benign = df[df['Prediction'] == 'BENIGN']  # Correctly filter BENIGN rows

        # Ensure they are saved separately
        ddos_csv = os.path.join('static', 'ddos_predictions.csv')
        benign_csv = os.path.join('static', 'benign_predictions.csv')

        df_ddos.to_csv(ddos_csv, index=False)  # Save only DDoS predictions
        df_benign.to_csv(benign_csv, index=False)  # Save only BENIGN predictions

        # Generate Pie and Bar charts
        labels = ['BENIGN', 'DDoS']
        sizes = [len(df_benign), len(df_ddos)]

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['blue', 'orange'])
        ax.set_title('Prediction Results')
        pie_chart_path = os.path.join('static', 'prediction_pie_chart.png')
        fig.savefig(pie_chart_path)
        plt.close(fig)

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(labels, sizes, color=['blue', 'orange'])
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Counts')
        bar_chart_path = os.path.join('static', 'prediction_bar_chart.png')
        fig.savefig(bar_chart_path)
        plt.close(fig)

        # Render the results page
        return render_template('results.html',
                               accuracy=round((np.sum(rf_pred == svm_pred) / len(rf_pred)) * 100, 2),
                               pie_chart='prediction_pie_chart.png',
                               bar_chart='prediction_bar_chart.png',
                               ddos_csv='ddos_predictions.csv',
                               benign_csv='benign_predictions.csv')

    return render_template('ddos.html')

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename, as_attachment=True)


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
