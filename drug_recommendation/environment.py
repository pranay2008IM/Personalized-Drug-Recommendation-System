import gym
import numpy as np
from gym import spaces
from data_processor import DataProcessor

class DrugRecommendationEnv(gym.Env):
    """Custom Environment for Drug Recommendation System"""
    def __init__(self):
        super(DrugRecommendationEnv, self).__init__()
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        # Get available drugs
        self.available_drugs = self.data_processor.get_available_drugs()
        self.n_drugs = len(self.available_drugs)
        
        # Calculate state size from preprocessed data
        sample_state = self.data_processor.process_patient_data(self.data_processor.patient_data.iloc[0])
        self.state_size = len(sample_state)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_drugs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_size,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Randomly select a patient
        patient_idx = np.random.randint(0, len(self.data_processor.patient_data))
        patient_data = self.data_processor.patient_data.iloc[patient_idx]
        
        # Process patient data
        self.state = self.data_processor.process_patient_data(patient_data)
        return self.state
    
    def step(self, action):
        """Execute action and return new state"""
        # Get drug and patient info
        drug_id = self.available_drugs[action]
        drug = self.data_processor.drug_data[self.data_processor.drug_data['drug_id'] == drug_id].iloc[0]
        
        # Get patient conditions
        condition_features = [f for f in self.data_processor.categorical_features if f.startswith('condition_')]
        patient_conditions = []
        for condition in condition_features:
            value = self.data_processor.patient_data.iloc[0][condition]
            if value != 'none':
                patient_conditions.append(value)
        
        # Calculate base reward from effectiveness
        reward = self.data_processor.get_drug_effectiveness(drug_id, self.state)
        
        # Boost reward for matching conditions
        if drug['primary_condition'] in patient_conditions:
            reward *= 2.0
        if drug['secondary_condition'] in patient_conditions:
            reward *= 1.5
        
        # Penalize for mismatched conditions
        if not any(cond in [drug['primary_condition'], drug['secondary_condition']] 
                  for cond in patient_conditions):
            reward *= 0.3
        
        # Penalize for contraindications
        contraindications = drug['contraindications'].split('|')
        if any(contra in patient_conditions for contra in contraindications):
            reward *= 0.1
        
        # Get patient allergies
        allergies = str(self.data_processor.patient_data.iloc[0]['allergies']).lower()
        if allergies != 'none' and allergies in drug['drug_name'].lower():
            reward *= 0.1
        
        # Consider side effects
        side_effects = drug['side_effects'].split('|')
        if len(side_effects) > 3:
            reward *= 0.9
        
        # Ensure reward is between 0 and 1
        reward = min(max(reward, 0.0), 1.0)
        
        # Move to next state (in this case, same state as it's a single-step environment)
        done = True
        info = {
            'drug_name': drug['drug_name'],
            'primary_condition': drug['primary_condition'],
            'secondary_condition': drug['secondary_condition'],
            'contraindications': drug['contraindications'],
            'side_effects': drug['side_effects']
        }
        
        return self.state, reward, done, info
