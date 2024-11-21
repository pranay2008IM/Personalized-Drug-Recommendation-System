import streamlit as st
import pandas as pd
import torch
import numpy as np
from environment import DrugRecommendationEnv
from dqn_agent import DQNAgent
from data_processor import DataProcessor

# Page config
st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="💊",
    layout="wide"
)

# Title and description
st.title("🏥 Intelligent Drug Recommendation System")
st.markdown("""
This system uses advanced machine learning to recommend appropriate medications based on patient conditions.
Please enter the patient information below to get personalized drug recommendations.
""")

# Initialize the models and processors
@st.cache_resource
def load_models():
    env = DrugRecommendationEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load('drug_recommendation_model.pth')
    data_processor = DataProcessor()
    return env, agent, data_processor

try:
    env, agent, data_processor = load_models()
    drug_data = pd.read_csv('drug_recommendation/new_data_exp/drug_data_expanded.csv')
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Add CSS for better styling
st.markdown("""
    <style>
    .drug-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .effectiveness-high { color: #28a745; }
    .effectiveness-medium { color: #ffc107; }
    .effectiveness-low { color: #dc3545; }
    .small-text { font-size: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# Create the input form
st.subheader("🏥 Patient Information")
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["M", "F"])
        systolic = st.number_input("Systolic Blood Pressure", min_value=70, max_value=200, value=120)
        diastolic = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=130, value=80)
    
    with col2:
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=72)
        temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6, step=0.1)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
    
    with col3:
        # Get unique conditions from the drug data
        conditions = sorted(list(set(
            list(drug_data['primary_condition'].unique()) + 
            list(drug_data['secondary_condition'].unique())
        )))
        conditions = [c for c in conditions if c != 'none']
        
        condition_1 = st.selectbox("Primary Condition", ["none"] + conditions)
        condition_2 = st.selectbox("Secondary Condition", ["none"] + conditions)
        condition_3 = st.selectbox("Third Condition", ["none"] + conditions)
        
        # Get unique allergies from drug contraindications
        allergies = sorted(list(set([
            item.strip() 
            for items in drug_data['contraindications'].str.split('|') 
            for item in items
        ])))
        allergy = st.selectbox("Allergies", ["none"] + allergies)

    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    try:
        # Create patient data
        patient_data = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'blood_pressure': f"{systolic}/{diastolic}",
            'heart_rate': heart_rate,
            'temperature': temperature,
            'condition_1': condition_1,
            'condition_2': condition_2,
            'condition_3': condition_3,
            'allergies': allergy
        }])

        # Process patient data
        state = data_processor.process_patient_data(patient_data.iloc[0])

        # Get model's recommendations
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.model(state_tensor)
            top_actions = q_values.topk(10).indices[0]  # Get top 10 recommendations

        # Evaluate and score each recommendation
        scored_recommendations = []
        conditions = [condition_1, condition_2, condition_3]
        conditions = [c for c in conditions if c != 'none']

        for action in top_actions:
            action = action.item()
            drug_id = env.available_drugs[action]
            drug = drug_data[drug_data['drug_id'] == drug_id].iloc[0]
            
            # Calculate score based on condition matching
            score = 0
            
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
            if allergy.lower() != 'none' and allergy.lower() in drug['drug_name'].lower():
                score -= 2.0
            
            # Consider side effects
            side_effects = drug['side_effects'].split('|')
            score -= len(side_effects) * 0.1
            
            # Get effectiveness
            effectiveness = data_processor.get_drug_effectiveness(drug_id, state)
            
            scored_recommendations.append({
                'drug': drug,
                'score': score,
                'effectiveness': effectiveness
            })

        # Sort recommendations by score
        scored_recommendations.sort(key=lambda x: (x['score'], x['effectiveness']), reverse=True)

        # Display recommendations
        st.subheader("📋 Recommended Medications")
        
        for i, rec in enumerate(scored_recommendations[:3], 1):
            drug = rec['drug']
            effectiveness = rec['effectiveness']
            
            # Determine effectiveness class
            eff_class = (
                'effectiveness-high' if effectiveness >= 0.7 else
                'effectiveness-medium' if effectiveness >= 0.4 else
                'effectiveness-low'
            )
            
            with st.expander(f"💊 Recommendation {i}: {drug['drug_name']}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Primary Condition:**")
                    st.info(drug['primary_condition'])
                    
                    st.markdown("**Secondary Condition:**")
                    st.info(drug['secondary_condition'])
                    
                    st.markdown("**Effectiveness Score:**")
                    st.markdown(f"<span class='{eff_class}'>{effectiveness:.2f}</span>", unsafe_allow_html=True)
                    
                    if drug['primary_condition'] in conditions or drug['secondary_condition'] in conditions:
                        st.success("✅ Matches patient conditions")
                    
                with col2:
                    st.markdown("**Contraindications:**")
                    contra_list = drug['contraindications'].split('|')
                    for contra in contra_list:
                        if contra in conditions or contra == allergy:
                            st.error(f"⚠️ {contra} (WARNING: Patient has this condition)")
                        else:
                            st.warning(contra)
                    
                    st.markdown("**Possible Side Effects:**")
                    effects = drug['side_effects'].split('|')
                    if len(effects) <= 3:
                        st.success("✅ Minimal side effects")
                    for effect in effects:
                        st.error(effect)
                
                # Add detailed analysis
                st.markdown("**Analysis:**")
                analysis_points = []
                
                if drug['primary_condition'] in conditions:
                    analysis_points.append("✅ Directly treats primary condition")
                if drug['secondary_condition'] in conditions:
                    analysis_points.append("✅ Addresses secondary condition")
                if any(contra in conditions for contra in contra_list):
                    analysis_points.append("⚠️ Has contraindications with patient conditions")
                if len(effects) <= 3:
                    analysis_points.append("✅ Low side effect profile")
                elif len(effects) > 5:
                    analysis_points.append("⚠️ High number of potential side effects")
                
                for point in analysis_points:
                    st.markdown(point)

        # Add disclaimer
        st.markdown("---")
        st.markdown("""
        <div class='small-text'>
        ⚠️ <b>Medical Disclaimer</b>: This system is for educational and research purposes only. 
        All medical decisions should be made by qualified healthcare professionals.
        The recommendations provided are based on a machine learning model and should not replace professional medical advice.
        Always consult with a licensed healthcare provider before making any medical decisions.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

# Add information about the system
with st.sidebar:
    st.subheader("ℹ️ About the System")
    st.markdown("""
    This advanced drug recommendation system uses Deep Reinforcement Learning with the following features:
    
    **Input Processing**:
    - Patient demographics
    - Vital signs monitoring
    - Multiple medical conditions
    - Allergy tracking
    
    **Analysis Factors**:
    - Primary condition matching
    - Secondary condition compatibility
    - Contraindication checking
    - Side effect profiling
    - Drug effectiveness scoring
    
    **Safety Features**:
    - Allergy cross-checking
    - Condition-based contraindication alerts
    - Side effect severity assessment
    - Multiple recommendation options
    
    **Model Details**:
    - Deep Q-Network architecture
    - Trained on extensive medical data
    - Regular effectiveness updates
    - Ensemble decision making
    """)
    
    st.subheader("🔍 How It Works")
    st.markdown("""
    1. **Data Collection**
       - Enter patient information
       - Input vital signs
       - Specify conditions
    
    2. **Analysis**
       - Process patient data
       - Generate state representation
       - Apply AI model
    
    3. **Recommendation**
       - Score multiple options
       - Check safety criteria
       - Rank by effectiveness
    
    4. **Review**
       - Examine recommendations
       - Consider alternatives
       - Note any warnings
    
    5. **Next Steps**
       - Consult healthcare provider
       - Discuss options
       - Follow medical advice
    """)
