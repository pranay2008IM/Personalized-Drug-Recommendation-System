import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Create directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'new_data_exp'), exist_ok=True)

# Lists for generating realistic data

# Save to CSV files
print("Saving files...")
output_dir = os.path.join(os.path.dirname(__file__), 'data', 'new_data_exp')
patient_df.to_csv(os.path.join(output_dir, 'patient_data_1000.csv'), index=False)
drug_df.to_csv(os.path.join(output_dir, 'drug_data_expanded.csv'), index=False)

print("Dataset generation complete!")
print(f"Generated {len(patient_df)} patient records")
print(f"Generated {len(drug_df)} drug records")
print(f"Files saved in: {output_dir}")