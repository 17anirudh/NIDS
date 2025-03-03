from flask import Flask, request, render_template
import joblib, pickle
import numpy as np
import pandas as pd

dt_path = 'models/decision_tree.pkl'
ada_path = 'models/adaboost.pkl'
cat_path = 'models/catboost.pkl'
gb_path = 'models/gradient_boosting.pkl'
rf_path = 'models/random_forest.pkl'
stack_path = 'models/stacking_classifier.pkl'
xgb_path = 'models/xgboost.pkl'
encoder_path = 'models/encoders.pkl'
scaler_path = 'models/scalers.pkl'
target_path = 'models/target.pkl'


def load_model(file_path):
    """
    Input: path of individual model

    Processing: joblib.load() and checks if that model has .predict()

    Output: returns the processed model
    """
    try:
        with open(file_path, 'rb') as file:
            model = joblib.load(file)
            if not hasattr(model, 'predict'):
                return None  
            return model
    except Exception:
        return None  


dt = load_model(dt_path)
ada = load_model(ada_path)
cat = load_model(cat_path)
gb = load_model(gb_path)
rf = load_model(rf_path)
stack = load_model(stack_path)
xgb = load_model(xgb_path)

# Load encoders and scalers 
with open(encoder_path, 'rb') as file:
    encoders = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scalers = pickle.load(file)

with open(target_path, 'rb') as file:
    target = pickle.load(file)


model_dict = {}
for name, model in [('dt', dt), ('rf', rf), ('ada', ada), ('stack', stack), ('xgb', xgb), ('cat', cat), ('gb', gb)]:
    if model is not None:
        model_dict[name] = model
    else:
        print(f"Warning: Skipping model '{name}' as it's None or doesn't have predict method")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create a DataFrame from form data
        input_data = {k: [v] for k, v in request.form.items() if k != 'model'}
        selected_model = request.form.get('model')
        input_df = pd.DataFrame(input_data)
        
        # Convert numeric columns to float
        numeric_cols = ['count', 'dst_host_diff_srv_rate', 'dst_host_rerror_rate', 
                       'serror_rate', 'dst_host_serror_rate', 'diff_srv_rate',
                       'srv_serror_rate', 'dst_host_count', 'dst_host_same_srv_rate']

        categorical_cols = ['protocol_type', 'service']
        
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(float)
            if col in scalers:
                input_df[col] = scalers[col].transform(input_df[[col]])
        
        # Apply encoders to categorical columns
        
        for col in categorical_cols:
            if col in input_df.columns and col in encoders:
                input_df[col] = encoders[col].transform(input_df[[col]])
        
        # Apply scalers to numeric columns
        for col in numeric_cols:
            if col in input_df.columns and col in scalers:
                input_df[col] = scalers[col].transform(input_df[[col]])
        
        # Convert DataFrame to numpy array for prediction
        features = input_df.values
        
        if selected_model in model_dict:
            model = model_dict[selected_model]
            if selected_model == 'xgb':
                import xgboost as xgb
                # Convert to DMatrix for XGBoost
                dmatrix = xgb.DMatrix(features)
                prediction = model.predict(dmatrix)
            else:
                # Regular prediction for other models
                prediction = model.predict(features)
            
            
            # Apply inverse_transform to the prediction
            if hasattr(target, 'inverse_transform'):
                output = target.inverse_transform([prediction[0]])[0]
            else:
                output = prediction[0]
                
            return render_template('index.html', prediction_text=f'Prediction: {output}')
        else:
            return render_template('index.html', prediction_text=f"Error: Invalid model selection. Available models: {list(model_dict.keys())}")
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)