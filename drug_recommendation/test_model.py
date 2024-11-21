import torch
import numpy as np
from environment import DrugRecommendationEnv
from dqn_agent import DQNAgent
from data_processor import DataProcessor
import pandas as pd
import os

def load_test_data(file_path):
    """Load patient data from test file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out comments and empty lines
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    if not data_lines:
        raise ValueError("No patient data found in the input file!")
    
    # Create DataFrame with proper columns
    columns = ['age', 'gender', 'blood_pressure', 'heart_rate', 'temperature', 
              'condition_1', 'condition_2', 'condition_3', 'allergies']
    
    # Parse each line into a list of values
    data = [line.split(',') for line in data_lines]
    return pd.DataFrame(data, columns=columns)

def test_model(model_path='drug_recommendation_model.pth', test_file='drug_recommendation/test_input_format.txt'):
    """Test the trained model with patient data from file"""
    try:
        # Load test data
        test_data = load_test_data(test_file)
        print(f"\nLoaded {len(test_data)} patient(s) from test file.")
        
        # Initialize environment and agent
        env = DrugRecommendationEnv()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        
        # Load trained model
        agent.load(model_path)
        print("Loaded trained model successfully.")
        
        # Get drug data for recommendations
        data_processor = DataProcessor()
        drug_data = pd.read_csv('drug_recommendation/new_data_exp/drug_data_expanded.csv')
        
        # Process each patient
        for idx, patient in test_data.iterrows():
            print(f"\nAnalyzing patient {idx + 1}:")
            print(f"Age: {patient['age']}, Gender: {patient['gender']}")
            print(f"Conditions: {patient['condition_1']}, {patient['condition_2']}, {patient['condition_3']}")
            print(f"Allergies: {patient['allergies']}")
            
            # Create temporary patient entry in data processor
            state = data_processor.process_patient_data(patient)
            
            # Get top N recommendations from model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.model(state_tensor)
                top_actions = q_values.topk(5).indices[0]
            
            # Evaluate each action with our criteria
            best_action = None
            best_score = -float('inf')
            best_drug = None
            
            for action in top_actions:
                action = action.item()
                drug_id = env.available_drugs[action]
                drug = drug_data[drug_data['drug_id'] == drug_id].iloc[0]
                
                # Calculate score based on condition matching
                score = 0
                conditions = [patient['condition_1'], patient['condition_2'], patient['condition_3']]
                conditions = [c for c in conditions if c != 'none']
                
                # Primary condition match
                if drug['primary_condition'] in conditions:
                    score += 2.0
                
                # Secondary condition match
                if drug['secondary_condition'] in conditions:
                    score += 1.0
                
                # Penalize for no condition match
                if not any(cond in [drug['primary_condition'], drug['secondary_condition']] 
                          for cond in conditions):
                    score -= 1.0
                
                # Check contraindications
                contraindications = drug['contraindications'].split('|')
                if any(contra in conditions for contra in contraindications):
                    score -= 2.0
                
                # Check allergies
                if str(patient['allergies']).lower() != 'none' and \
                   str(patient['allergies']).lower() in drug['drug_name'].lower():
                    score -= 2.0
                
                # Consider side effects
                side_effects = drug['side_effects'].split('|')
                score -= len(side_effects) * 0.1
                
                # Update best action if this score is higher
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_drug = drug
            
            # Get effectiveness for best drug
            effectiveness = data_processor.get_drug_effectiveness(
                env.available_drugs[best_action], state)
            
            print("\nRecommended Medication:")
            print(f"Drug: {best_drug['drug_name']}")
            print(f"Primary Condition: {best_drug['primary_condition']}")
            print(f"Secondary Condition: {best_drug['secondary_condition']}")
            print(f"Predicted Effectiveness: {effectiveness:.2f}")
            print(f"Contraindications: {best_drug['contraindications']}")
            print(f"Possible Side Effects: {best_drug['side_effects']}")
            
            print("\n" + "="*50)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease make sure:")
        print("1. The input file exists and follows the correct format")
        print("2. The trained model file exists")
        print("3. All required data files are present")

if __name__ == "__main__":
    # You can specify different test file if needed
    test_model()
