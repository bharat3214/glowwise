import pandas as pd
import numpy as np
import random
import os

def generate_hair_dataset(num_samples=5000):
    np.random.seed(123)
    random.seed(123)

    genders = ['Male', 'Female']
    scalp_types = ['Oily', 'Dry', 'Normal']
    textures = ['Straight', 'Wavy', 'Curly', 'Coily']
    thicknesses = ['Fine', 'Medium', 'Thick']
    environments = ['Humid', 'Dry', 'Moderate']
    
    # Binary conditions
    dandruff_options = ['Yes', 'No']
    hairfall_options = ['Yes', 'No']
    frizz_options = ['Yes', 'No']
    color_treated_options = ['Yes', 'No']

    data = []

    for _ in range(num_samples):
        gender = random.choice(genders)
        scalp = random.choice(scalp_types)
        texture = random.choice(textures)
        thickness = random.choice(thicknesses)
        env = random.choice(environments)
        
        dandruff = random.choice(dandruff_options)
        hairfall = random.choice(hairfall_options)
        frizz = random.choice(frizz_options)
        color = random.choice(color_treated_options)

        # Logic for target variable 'Haircare_Routine'
        if dandruff == 'Yes':
            routine = "Anti-Dandruff"
        elif hairfall == 'Yes':
            routine = "Hairfall Control"
        elif color == 'Yes':
            routine = "Color Protect"
        elif frizz == 'Yes' or scalp == 'Dry' or texture in ['Curly', 'Coily']:
            routine = "Frizz Control & Moisture"
        elif thickness == 'Fine' or scalp == 'Oily':
            routine = "Daily Volume"
        else:
            routine = np.random.choice(["Daily Volume", "Frizz Control & Moisture"], p=[0.5, 0.5])

        row = {
            'Gender': gender,
            'Scalp_Type': scalp,
            'Texture': texture,
            'Thickness': thickness,
            'Environment': env,
            'Dandruff': dandruff,
            'Hairfall': hairfall,
            'Frizz': frizz,
            'Color_Treated': color,
            'Routine': routine
        }
        data.append(row)

    df = pd.DataFrame(data)
    
    os.makedirs('data', exist_ok=True)
    out_path = os.path.join('data', 'haircare_dataset.csv')
    df.to_csv(out_path, index=False)
    print(f"Haircare Dataset generated successfully with {num_samples} rows at {out_path}")

if __name__ == "__main__":
    generate_hair_dataset()
