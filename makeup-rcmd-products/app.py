import streamlit as st
import json
from src.ml_engine import MLEngine
from src.groq_verifier import GroqVerifier

def load_db(path):
    with open(path, 'r') as f:
        return json.load(f)

def show_products(routine, db, features, groq_verifier):
    products = db.get(routine, [])
    if products:
        filtered_products = []
        type_count = {}
        for p in products:
            t = p['type']
            type_count[t] = type_count.get(t, 0) + 1
            if type_count[t] <= 2:
                filtered_products.append(p)

        st.markdown("### 🧴 Recommended Products")
        for prod in filtered_products:
            with st.expander(f"✅ {prod['type']}: {prod['name']}"):
                st.write(f"**Ingredients:** {prod['ingredients']}")
                if st.button("Check if this is suitable", key=f"btn_rec_{prod['name']}"):
                    st.session_state['verify_request'] = {'profile': features, 'product_name': prod['name'], 'ingredients': prod['ingredients']}

        st.markdown("### 🚫 Products to Avoid")
        other_routines = [r for r in db.keys() if r != routine]
        if other_routines:
            avoid_routine = other_routines[0]
            avoid_products = db[avoid_routine][:2]
            for prod in avoid_products:
                with st.expander(f"❌ {prod['type']}: {prod['name']}"):
                    st.write(f"**Ingredients:** {prod['ingredients']}")
                    if st.button("Why is this not suitable?", key=f"btn_avoid_{prod['name']}"):
                        st.session_state['verify_request'] = {'profile': features, 'product_name': prod['name'], 'ingredients': prod['ingredients']}

def run_app():
    st.set_page_config(page_title="AI Beauty Recommender", layout="wide")
    st.title("✨ AI Beauty Recommendation System")
    st.markdown("Discover perfect Skincare and Haircare routines based on your unique profile using Machine Learning.")

    with st.expander("📊 View ML Training Datasets"):
        try:
            import pandas as pd
            colA, colB = st.columns(2)
            with colA:
                st.write("**Skincare Dataset**")
                st.dataframe(pd.read_csv('data/skincare_dataset.csv').head(50))
            with colB:
                st.write("**Haircare Dataset**")
                st.dataframe(pd.read_csv('data/haircare_dataset.csv').head(50))
        except Exception:
            st.error("Datasets not found. Ensure scripts have been run.")

    try:
        ml_skin = MLEngine()
        ml_hair = MLEngine(model_path='models/hair_model.pkl', encoder_path='models/hair_encoders.pkl')
        db_skin = load_db('data/product_db.json')
        db_hair = load_db('data/hair_product_db.json')
        groq_verifier = GroqVerifier()
    except Exception as e:
        st.error(f"Error initializing system APIs and models: {str(e)}")
        return

    # -------- SKINCARE SECTION --------
    with st.expander("🧴 Skincare Recommendation System", expanded=True):
        st.subheader("📋 Your Skin Profile")
        with st.form("skin_form"):
            col1, col2 = st.columns(2)
            with col1:
                s_gender = st.radio("Gender", options=["Female", "Male"], key='s_gender')
                skin_type = st.selectbox("Skin Type", options=["Oily", "Dry", "Combination", "Sensitive"])
                sensitivity = st.selectbox("Sensitivity", options=["Low", "Medium", "High"])
                environment = st.selectbox("Environment", options=["Humid", "Dry", "Moderate"])
            with col2:
                age_group = st.selectbox("Age Group", options=["Teen", "Adult", "Mature"])
                pores = st.selectbox("Pore Size", options=["Small", "Medium", "Large"])
                preference = st.selectbox("Routine Length", options=["Minimal", "Standard", "Advanced"])
                
                st.markdown("**Conditions**")
                has_acne = st.checkbox("Acne-prone (Breakouts)")
                has_pigmentation = st.checkbox("Pigmentation / Dark Spots")
                has_redness = st.checkbox("Redness / Rosacea")
                has_wrinkles = st.checkbox("Fine Lines / Wrinkles")
                has_blackheads = st.checkbox("Blackheads / Whiteheads")

            skin_submit = st.form_submit_button("Analyze My Skin")

        if skin_submit:
            st.session_state['skin_profile'] = {
                'Gender': s_gender, 'Skin_Type': skin_type,
                'Acne': 'Yes' if has_acne else 'No', 'Pigmentation': 'Yes' if has_pigmentation else 'No',
                'Redness': 'Yes' if has_redness else 'No', 'Wrinkles': 'Yes' if has_wrinkles else 'No',
                'Blackheads': 'Yes' if has_blackheads else 'No',
                'Sensitivity': sensitivity, 'Pores': pores,
                'Environment': environment, 'Age_Group': age_group, 'Preference': preference
            }

        if 'skin_profile' in st.session_state:
            features = st.session_state['skin_profile']
            routine = ml_skin.predict_routine(features)
            exp = ml_skin.explain_prediction(features)
            
            st.success("✅ Analysis Complete!")
            st.subheader(f"🔮 Recommended Routine: **{routine}**")

            with st.expander("📊 Why was this recommended? (Explainable AI)"):
                if exp:
                    import pandas as pd
                    df_exp = pd.DataFrame({"Feature Impact": list(exp.values())}, index=list(exp.keys()))
                    st.bar_chart(df_exp)
            
            show_products(routine, db_skin, features, groq_verifier)

    # -------- HAIRCARE SECTION --------
    with st.expander("💆‍♀️ Haircare Recommendation System"):
        st.subheader("📋 Your Hair Profile")
        with st.form("hair_form"):
            col1, col2 = st.columns(2)
            with col1:
                h_gender = st.radio("Gender", options=["Female", "Male"], key='h_gender')
                scalp = st.selectbox("Scalp Type", options=["Oily", "Dry", "Normal"])
                texture = st.selectbox("Hair Texture", options=["Straight", "Wavy", "Curly", "Coily"])
                thickness = st.selectbox("Hair Thickness", options=["Fine", "Medium", "Thick"])
            with col2:
                h_env = st.selectbox("Environment", options=["Humid", "Dry", "Moderate"], key='h_env')
                st.markdown("**Conditions / Concerns **")
                has_dandruff = st.checkbox("Dandruff / Flaky Scalp")
                has_hairfall = st.checkbox("Hairfall / Thinning")
                has_frizz = st.checkbox("Frizz / Damage")
                is_color = st.checkbox("Color-Treated / Bleached")
                
            hair_submit = st.form_submit_button("Analyze My Hair")

        if hair_submit:
            st.session_state['hair_profile'] = {
                'Gender': h_gender, 'Scalp_Type': scalp,
                'Texture': texture, 'Thickness': thickness, 
                'Environment': h_env,
                'Dandruff': 'Yes' if has_dandruff else 'No',
                'Hairfall': 'Yes' if has_hairfall else 'No',
                'Frizz': 'Yes' if has_frizz else 'No',
                'Color_Treated': 'Yes' if is_color else 'No'
            }

        if 'hair_profile' in st.session_state:
            features = st.session_state['hair_profile']
            routine = ml_hair.predict_routine(features)
            exp = ml_hair.explain_prediction(features)
            
            st.success("✅ Analysis Complete!")
            st.subheader(f"🔮 Recommended Routine: **{routine}**")

            with st.expander("📊 Why was this recommended? (Explainable AI)"):
                if exp:
                    import pandas as pd
                    df_exp = pd.DataFrame({"Feature Impact": list(exp.values())}, index=list(exp.keys()))
                    st.bar_chart(df_exp)
            
            show_products(routine, db_hair, features, groq_verifier)

    # -------- GROQ ENGINE SYSTEM (GLOBALS) --------
    if 'verify_request' in st.session_state:
        req = st.session_state['verify_request']
        st.markdown("---")
        st.subheader(f"🧠 AI Verification: {req['product_name']}")
        with st.spinner("Consulting Groq API for analysis..."):
            analysis = groq_verifier.verify_product(req['profile'], req['product_name'], req['ingredients'])
            st.info(analysis)
            
        if st.button("Clear Check"):
            del st.session_state['verify_request']
            st.rerun()
            
    st.markdown("---")
    st.subheader("🔍 Search & Verify Any Product")
    st.markdown("Want to check a specific product not listed above? Enter it here to let AI analyze it based on your most recently submitted profile.")
    
    custom_product = st.text_input("Enter brand and product name (e.g., Olaplex No.3 or Cerave Moisturizer)")
    if st.button("Verify Custom Product"):
        if custom_product.strip():
            with st.spinner("Asking AI Dermatologist / Trichologist..."):
                profile = st.session_state.get('skin_profile', st.session_state.get('hair_profile', {}))
                analysis = groq_verifier.verify_product(profile, custom_product, "")
                st.info(analysis)
        else:
            st.warning("Please enter a product name.")

if __name__ == "__main__":
    run_app()
