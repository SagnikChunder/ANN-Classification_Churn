import gradio as gr
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load model and preprocessors
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    onehotencoder_geo = pickle.load(file)

def predict_churn(geography, gender, age, balance, credit_score, estimated_salary, tenure, num_of_products, has_cr_card, is_active_member):
    """
    Predict customer churn based on input features
    """
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatesSalary': [estimated_salary]
    })

    # One-hot encode geography
    geo_encoded = onehotencoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Predict Churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    
    # Format output
    probability_text = f"Churn Probability: {prediction_proba:.2f}"
    
    if prediction_proba > 0.5:
        result_text = "The customer is likely to churn."
        risk_level = "High Risk"
    else:
        result_text = "The customer is not likely to churn."
        risk_level = "Low Risk"
    
    return probability_text, result_text, risk_level

# Create Gradio interface
with gr.Blocks(title="Customer Churn Prediction") as demo:
    gr.Markdown("# Customer Churn Prediction")
    gr.Markdown("Enter customer details to predict the likelihood of churn.")
    
    with gr.Row():
        with gr.Column():
            geography = gr.Dropdown(
                choices=list(onehotencoder_geo.categories_[0]), 
                label="Geography",
                value=onehotencoder_geo.categories_[0][0]
            )
            gender = gr.Dropdown(
                choices=list(label_encoder_gender.classes_), 
                label="Gender",
                value=label_encoder_gender.classes_[0]
            )
            age = gr.Slider(
                minimum=18, 
                maximum=92, 
                value=35, 
                label="Age"
            )
            balance = gr.Number(
                label="Balance", 
                value=0
            )
            credit_score = gr.Number(
                label="Credit Score", 
                value=600
            )
        
        with gr.Column():
            estimated_salary = gr.Number(
                label="Estimated Salary", 
                value=50000
            )
            tenure = gr.Slider(
                minimum=0, 
                maximum=10, 
                value=5, 
                label="Tenure"
            )
            num_of_products = gr.Slider(
                minimum=1, 
                maximum=4, 
                value=2, 
                label="Number of Products"
            )
            has_cr_card = gr.Dropdown(
                choices=[0, 1], 
                label="Has Credit Card", 
                value=1
            )
            is_active_member = gr.Dropdown(
                choices=[0, 1], 
                label="Is Active Member", 
                value=1
            )
    
    predict_btn = gr.Button("Predict Churn", variant="primary")
    
    with gr.Row():
        with gr.Column():
            probability_output = gr.Textbox(label="Churn Probability", interactive=False)
            result_output = gr.Textbox(label="Prediction Result", interactive=False)
            risk_output = gr.Textbox(label="Risk Level", interactive=False)
    
    # Set up the prediction function
    predict_btn.click(
        fn=predict_churn,
        inputs=[
            geography, gender, age, balance, credit_score, 
            estimated_salary, tenure, num_of_products, 
            has_cr_card, is_active_member
        ],
        outputs=[probability_output, result_output, risk_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
