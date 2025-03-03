from flask import Flask, request, render_template
import joblib, pickle
import numpy as np
import pandas as pd
import xgboost as xgb

dt_path = 'models/decision_tree.pkl'
ada_path = 'models/adaboost.pkl'
nb_path = 'models/naive_bayes.pkl'
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
nb = load_model(nb_path)
rf = load_model(rf_path)
stack = load_model(stack_path)

xgb_clf = xgb.Booster()
xgb_clf.load_model("models/xgb.bin")

# Load encoders and scalers 
with open(encoder_path, 'rb') as file:
    encoders = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scalers = pickle.load(file)

with open(target_path, 'rb') as file:
    target = pickle.load(file)


model_dict = {}
for name, model in [('dt', dt), ('nb', nb), ('rf', rf), ('ada', ada), ('stack', stack), ('xgb', xgb_clf), ('cat', cat), ('gb', gb)]:
    if model is not None:
        model_dict[name] = model

app = Flask(__name__)

@app.route('/')
def home():
    """
    Output: returns index.html
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        """
        Input: When user clicks 'predict', the form values

        Processing: Encodes or scales respectively and predicts from already trained models imported

        Output: Predicted value using inverse label transform
        """

        input_data = {k: [v] for k, v in request.form.items() if k != 'model'}
        selected_model = request.form.get('model')
        input_df = pd.DataFrame(input_data)
        expected_features = xgb_clf.feature_names
        print(expected_features)
        
        numeric_cols = ['count', 'dst_host_diff_srv_rate', 'dst_host_rerror_rate', 
                       'serror_rate', 'dst_host_serror_rate', 'diff_srv_rate',
                       'srv_serror_rate', 'dst_host_count', 'dst_host_same_srv_rate']
        categorical_cols = ['protocol_type', 'service']
        
        for col in input_df.columns:
            if col in scalers and numeric_cols:
                input_df[col] = input_df[col].astype(float)
                input_df[col] = scalers[col].transform(input_df[[col]])
            if col in categorical_cols and encoders:
                input_df[col] = encoders[col].transform(input_df[[col]])
        
        # Convert DataFrame to numpy array for prediction
        input_df = input_df[expected_features]
        features = input_df.values
        
        if selected_model in model_dict:
            model = model_dict[selected_model]
            if selected_model == 'xgb':
                dmatrix = xgb.DMatrix(features, feature_names=input_df.columns.tolist())
                prediction = np.round(model.predict(dmatrix)).astype(int)
            else:
                prediction = model.predict(features)
            

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