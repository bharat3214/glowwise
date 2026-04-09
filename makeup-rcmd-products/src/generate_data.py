import pandas as pd
import numpy as np
import random
import os

def generate_dataset(num_samples=5000):
    np.random.seed(42)
    random.seed(42)

    genders = ['Male', 'Female']
    skin_types = ['Oily', 'Dry', 'Combination', 'Sensitive']
    acne_options = ['Yes', 'No']
    pigmentation_options = ['Yes', 'No']
    redness_options = ['Yes', 'No']
    wrinkles_options = ['Yes', 'No']
    blackheads_options = ['Yes', 'No']
    sensitivities = ['Low', 'Medium', 'High']
    pores = ['Small', 'Medium', 'Large']
    environments = ['Humid', 'Dry', 'Moderate']
    age_groups = ['Teen', 'Adult', 'Mature']
    preferences = ['Minimal', 'Standard', 'Advanced']

    data = []

    for _ in range(num_samples):
        gender = random.choice(genders)
        skin_type = random.choice(skin_types)
        acne = random.choice(acne_options)
        pigmentation = random.choice(pigmentation_options)
        redness = random.choice(redness_options)
        wrinkles = random.choice(wrinkles_options)
        blackheads = random.choice(blackheads_options)
        
        if skin_type == 'Sensitive':
            sensitivity = np.random.choice(sensitivities, p=[0.1, 0.3, 0.6])
        else:
            sensitivity = np.random.choice(sensitivities, p=[0.6, 0.3, 0.1])
            
        pore_size = random.choice(pores)
        env = random.choice(environments)
        age = random.choice(age_groups)
        pref = random.choice(preferences)

        # Updated targets based on new multi-optons
        if (acne == 'Yes' or blackheads == 'Yes') and skin_type in ['Oily', 'Combination']:
            routine = "Acne Control"
        elif sensitivity == 'High' or skin_type == 'Sensitive' or redness == 'Yes':
            routine = "Sensitive Care"
        elif (age == 'Mature' or wrinkles == 'Yes') and acne == 'No':
            routine = "Anti-Aging"
        elif env == 'Dry' or skin_type == 'Dry':
            routine = "Hydration"
        elif skin_type == 'Oily' and env == 'Humid':
            routine = "Acne Control"
        else:
            routine = np.random.choice(["Hydration", "Anti-Aging", "Sensitive Care"], p=[0.6, 0.2, 0.2])

        row = {
            'Gender': gender,
            'Skin_Type': skin_type,
            'Acne': acne,
            'Pigmentation': pigmentation,
            'Redness': redness,
            'Wrinkles': wrinkles,
            'Blackheads': blackheads,
            'Sensitivity': sensitivity,
            'Pores': pore_size,
            'Environment': env,
            'Age_Group': age,
            'Preference': pref,
            'Routine': routine
        }
        data.append(row)

    df = pd.DataFrame(data)
    
    os.makedirs('data', exist_ok=True)
    out_path = os.path.join('data', 'skincare_dataset.csv')
    df.to_csv(out_path, index=False)
    print(f"Dataset generated successfully with {num_samples} rows at {out_path}")

if __name__ == "__main__":
    generate_dataset()
