import json
import torch
from model import create_model

def model_fn(model_dir):
    model = create_model(None)  # Create a blank model instance
    model.load(f"{model_dir}/dqn_trade_model.zip")  # Load the trained weights
    return model

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    obs = [0, 0, input_data['shares'], 0, 0]  # Example observation
    action, _ = model.predict(obs)
    return {'action': action, 'timestamp': input_data['time_horizon']}

def output_fn(prediction, content_type):
    return json.dumps(prediction), content_type
