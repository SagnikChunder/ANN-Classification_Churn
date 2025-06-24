import gradio as gr
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load model and preprocessors
# Try different loading methods to handle version compatibility
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    print(f"Error loading with standard method: {e}")
    try:
        # Try loading with compile=False to avoid compilation issues
        model = tf.keras.models.load_model('model.h5', compile=False)
        print("Model loaded successfully with compile=False")
    except Exception as e2:
        print(f"Error loading with compile=False: {e2}")
        try:
            # Try loading with custom_objects to handle deprecated parameters
            model = tf.keras.models.load_model('model.h5', custom_objects={}, compile=False)
            print("Model loaded successfully with custom_objects")
        except Exception as e3:
            print(f"Error loading .h5 file failed. Trying SavedModel format...")
            try:
                # Try loading SavedModel format if available
                model = tf.keras.models.load_model('model_savedmodel')
                print("Model loaded successfully from SavedModel format")
            except Exception as e4:
                print(f"All loading methods failed: {e4}")
                raise e4

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

def predict_churn(geography, gender, age, balance, credit_score, estimated_salary, tenure, num_of_products, has_cr_card, is_active_member):
    """
    Predict customer churn based on input features
    """
    try:
        print(f"Input values: geography={geography}, gender={gender}, age={age}")
        print(f"Balance={balance}, credit_score={credit_score}, salary={estimated_salary}")
        print(f"Tenure={tenure}, products={num_of_products}, card={has_cr_card}, active={is_active_member}")
        
        # Convert inputs to proper types
        age = int(age)
        balance = float(balance) if balance is not None else 0.0
        credit_score = float(credit_score) if credit_score is not None else 600.0
        estimated_salary = float(estimated_salary) if estimated_salary is not None else 50000.0
        tenure = int(tenure)
        num_of_products = int(num_of_products)
        has_cr_card = int(has_cr_card)
        is_active_member = int(is_active_member)
        
        # Validate gender input
        if gender not in label_encoder_gender.classes_:
            return f"Error: Invalid gender '{gender}'", "Please select a valid gender", "Error"
        
        # Validate geography input  
        if geography not in onehot_encoder_geo.categories_[0]:
            return f"Error: Invalid geography '{geography}'", "Please select a valid geography", "Error"
        
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
            'EstimatedSalary': [estimated_salary]  # Fixed typo: EstimatesSalary -> EstimatedSalary
        })
        
        print(f"Input data shape: {input_data.shape}")
        print(f"Input data:\n{input_data}")

        # One-hot encode geography
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        
        print(f"Geography encoded shape: {geo_encoded_df.shape}")

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        print(f"Combined data shape: {input_data.shape}")
        print(f"Combined data columns: {input_data.columns.tolist()}")
        
        # Scale input data
        input_data_scaled = scaler.transform(input_data)
        print(f"Scaled data shape: {input_data_scaled.shape}")

        # Predict Churn
        prediction = model.predict(input_data_scaled, verbose=0)
        prediction_proba = float(prediction[0][0])
        
        print(f"Raw prediction: {prediction}")
        print(f"Prediction probability: {prediction_proba}")
        
        # Format output
        probability_text = f"Churn Probability: {prediction_proba:.2f}"
        
        if prediction_proba > 0.5:
            result_text = "The customer is likely to churn."
            risk_level = "High Risk"
        else:
            result_text = "The customer is not likely to churn."
            risk_level = "Low Risk"
        
        return probability_text, result_text, risk_level
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, "Prediction failed", "Error"

# Create Gradio interface
with gr.Blocks(title="Customer Churn Prediction") as demo:
    gr.Markdown("# Customer Churn Prediction")
    gr.Markdown("Enter customer details to predict the likelihood of churn.")
    
    with gr.Row():
        with gr.Column():
            geography = gr.Dropdown(
                choices=list(onehot_encoder_geo.categories_[0]), 
                label="Geography",
                value=onehot_encoder_geo.categories_[0][0]
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
