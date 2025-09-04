from flask import Flask, render_template, request, jsonify
import numpy as np
from house_price_ml import preprocess_data, load_housing_data, zscore_normalize_features, predict

app = Flask(__name__)

# Global variables to store model and preprocessing data
model_params = None
feature_names = None
mu = None
sigma = None
test_data = None  # Store test data for visualization

def load_model():
    """Load the trained model parameters"""
    global model_params, feature_names, mu, sigma, test_data
    
    # Train the model if not already trained
    if model_params is None:
        print("Training model...")
        from house_price_ml import gradient_descent, split_data
        
        # Load and preprocess data
        data = load_housing_data()
        X, y, feature_names = preprocess_data(data)
        
        # Split and normalize data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        X_train_norm, mu, sigma = zscore_normalize_features(X_train)
        
        # Train model
        initial_w = np.zeros(X_train_norm.shape[1])
        initial_b = 0
        alpha = 0.01
        iterations = 1000
        
        w_final, b_final, J_history, _ = gradient_descent(
            X_train_norm, y_train, initial_w, initial_b, alpha, iterations
        )
        
        # Make predictions on test set for visualization
        X_test_norm = (X_test - mu) / sigma
        test_predictions = predict(X_test_norm, w_final, b_final)
        
        model_params = {
            'weights': w_final,
            'bias': b_final,
            'mu': mu,
            'sigma': sigma,
            'feature_names': feature_names
        }
        
        # Store test data for visualization
        test_data = {
            'y_actual': y_test,
            'y_predicted': test_predictions,
            'cost_history': J_history
        }
        
        print("Model trained successfully!")
        print(f"Features: {feature_names}")

def preprocess_user_input(form_data):
    """Convert user form input to model-ready format"""
    # Create a dict with default values
    input_dict = {
        'GrLivArea': float(form_data.get('living_area', 1500)),
        'TotalBsmtSF': float(form_data.get('basement_area', 1000)),
        '1stFlrSF': float(form_data.get('first_floor', 1000)),
        '2ndFlrSF': float(form_data.get('second_floor', 800)),
        'OverallQual': int(form_data.get('overall_quality', 6)),
        'OverallCond': int(form_data.get('overall_condition', 5)),
        'YearBuilt': int(form_data.get('year_built', 2000)),
        'GarageCars': int(form_data.get('garage_cars', 2)),
        'KitchenQual': int(form_data.get('kitchen_quality', 3)),
        'CentralAir': int(form_data.get('central_air', 1)),
        'Zone_C (all)': 0,
        'Zone_FV': 0,
        'Zone_RH': 0,
        'Zone_RL': 1,  # Default to most common zone
        'Zone_RM': 0,
        'Neighborhood_Popular': int(form_data.get('popular_neighborhood', 1))
    }
    
    # Handle zoning
    zone = form_data.get('zoning', 'RL')
    for zone_col in ['Zone_C (all)', 'Zone_FV', 'Zone_RH', 'Zone_RL', 'Zone_RM']:
        input_dict[zone_col] = 0
    if f'Zone_{zone}' in input_dict:
        input_dict[f'Zone_{zone}'] = 1
    
    # Convert to numpy array in the correct order
    feature_vector = np.array([input_dict[feature] for feature in feature_names])
    
    return feature_vector.reshape(1, -1)

@app.route('/')
def home():
    """Home page with input form"""
    load_model()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_price():
    """Handle prediction request"""
    try:
        load_model()
        
        # Get form data
        form_data = request.form.to_dict()
        
        # Preprocess input
        user_input = preprocess_user_input(form_data)
        
        # Normalize using training statistics
        user_input_norm = (user_input - mu) / sigma
        
        # Make prediction
        prediction = predict(user_input_norm, model_params['weights'], model_params['bias'])
        predicted_price = float(prediction[0])
        
        # Format the response
        result = {
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': f"${predicted_price:,.0f}",
            'input_summary': {
                'Living Area': f"{form_data.get('living_area', 1500)} sq ft",
                'Overall Quality': f"{form_data.get('overall_quality', 6)}/10",
                'Year Built': form_data.get('year_built', 2000),
                'Garage Cars': form_data.get('garage_cars', 2)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/visualization')
def visualization():
    load_model()
    return render_template('visualization.html')

@app.route('/get_plot_data')
def get_plot_data():
    load_model()
    
    if test_data is None:
        return jsonify({'error': 'No test data available'})
    
    # Prediction vs Actual scatter plot
    prediction_plot = {
        'x': [float(x) for x in test_data['y_actual'].tolist()],
        'y': [float(y) for y in test_data['y_predicted'].tolist()],
        'mode': 'markers',
        'type': 'scatter',
        'name': 'Predictions',
        'marker': {
            'size': 8,
            'color': 'rgba(55, 126, 184, 0.7)',
            'line': {'color': 'rgba(55, 126, 184, 1)', 'width': 1}
        }
    }
    
    # Perfect prediction line (y = x)
    min_price = float(min(min(test_data['y_actual']), min(test_data['y_predicted'])))
    max_price = float(max(max(test_data['y_actual']), max(test_data['y_predicted'])))
    
    perfect_line = {
        'x': [min_price, max_price],
        'y': [min_price, max_price],
        'mode': 'lines',
        'type': 'scatter',
        'name': 'Perfect Prediction',
        'line': {'color': 'red', 'dash': 'dash', 'width': 2}
    }
    
    # Cost history plot
    cost_plot = {
        'x': list(range(len(test_data['cost_history']))),
        'y': [float(cost) for cost in test_data['cost_history']],
        'mode': 'lines',
        'type': 'scatter',
        'name': 'Training Cost',
        'line': {'color': 'green', 'width': 3}
    }
    
    # Calculate R² and RMSE for display
    y_actual = test_data['y_actual']
    y_pred = test_data['y_predicted']
    
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    
    return jsonify({
        'prediction_plot': prediction_plot,
        'perfect_line': perfect_line,
        'cost_plot': cost_plot,
        'metrics': {
            'r2': float(round(r2, 3)),
            'rmse': float(round(rmse, 2)),
            'samples': int(len(y_actual))
        }
    })

if __name__ == '__main__':
    app.run()