import streamlit as st
import json
from src.ml_engine import MLEngine
from src.groq_verifier import GroqVerifier

def load_product_db():
    with open('data/product_db.json', 'r') as f:
        return json.load(f)

def run_app():
    st.set_page_config(page_title="AI Skincare Recommender", layout="wide")
    st.title("✨ AI Skincare Recommendation System")
    st.markdown("Discover the perfect skincare routine based on your unique skin profile.")

    with st.expander("📊 View ML Training Dataset"):
        try:
            import pandas as pd
            df = pd.read_csv('data/skincare_dataset.csv')
            st.dataframe(df)
        except Exception:
            st.error("Dataset not found. Please ensure it was generated.")

    try:
        ml_engine = MLEngine()
        product_db = load_product_db()
        groq_verifier = GroqVerifier()
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.warning("Please ensure the ML model is trained and 'data/product_db.json' exists.")
        return

    # Questionnaire
    with st.form("questionnaire"):
        st.subheader("📋 Your Skin Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.radio("Gender", options=["Female", "Male"])
            skin_type = st.selectbox("Skin Type", options=["Oily", "Dry", "Combination", "Sensitive"])
            sensitivity = st.selectbox("Sensitivity Level", options=["Low", "Medium", "High"])
            environment = st.selectbox("Environment", options=["Humid", "Dry", "Moderate"])
            
        with col2:
            age_group = st.selectbox("Age Group", options=["Teen", "Adult", "Mature"])
            pores = st.selectbox("Pore Size", options=["Small", "Medium", "Large"])
            preference = st.selectbox("Preferred Routine", options=["Minimal", "Standard", "Advanced"])
            
            st.markdown("**Skin Conditions (Select all that apply)**")
            has_acne = st.checkbox("Acne-prone (Breakouts)")
            has_pigmentation = st.checkbox("Pigmentation / Dark Spots")
            has_redness = st.checkbox("Redness / Rosacea")
            has_wrinkles = st.checkbox("Fine Lines / Wrinkles")
            has_blackheads = st.checkbox("Blackheads / Whiteheads")

        submitted = st.form_submit_button("Analyze My Skin")

    if submitted:
        # Saving submitted profile into session state to maintain it when verification buttons are clicked
        st.session_state['user_profile'] = {
            'Gender': gender,
            'Skin_Type': skin_type,
            'Acne': 'Yes' if has_acne else 'No',
            'Pigmentation': 'Yes' if has_pigmentation else 'No',
            'Redness': 'Yes' if has_redness else 'No',
            'Wrinkles': 'Yes' if has_wrinkles else 'No',
            'Blackheads': 'Yes' if has_blackheads else 'No',
            'Sensitivity': sensitivity,
            'Pores': pores,
            'Environment': environment,
            'Age_Group': age_group,
            'Preference': preference
        }

    if 'user_profile' in st.session_state:
        features = st.session_state['user_profile']
        
        with st.spinner("Analyzing profile using Machine Learning..."):
            routine = ml_engine.predict_routine(features)
            explanation = ml_engine.explain_prediction(features)

        st.success("✅ Analysis Complete!")
        st.subheader(f"🔮 Recommended Routine: **{routine}**")

        with st.expander("📊 Why was this recommended? (Explainable AI)"):
            st.markdown(f"These are the specific input features that mathematically influenced the **Machine Learning** model to assign you the `{routine}` routine:")
            
            if explanation:
                import pandas as pd
                df_exp = pd.DataFrame({
                    "Feature Impact": list(explanation.values())
                }, index=list(explanation.keys()))
                
                st.bar_chart(df_exp)
                st.markdown("*Positive bars (Right) drove the model toward this routine. Negative bars (Left) pushed the model away from this routine.*")
            else:
                st.write("Unable to extract feature significance.")

        products = product_db.get(routine, [])
        if products:
            # Max 2 per product type/category
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
                    if st.button("Check if this product is suitable", key=f"btn_rec_{prod['name']}"):
                        st.session_state['verify_request'] = {
                            'profile': features,
                            'product_name': prod['name'],
                            'ingredients': prod['ingredients']
                        }

            st.markdown("### 🚫 Products to Avoid")
            st.info("These products contain ingredients that clash with your routine and may cause irritation for your profile.")
            
            other_routines = [r for r in product_db.keys() if r != routine]
            if other_routines:
                avoid_routine = other_routines[0]
                avoid_products = product_db[avoid_routine][:2]
                
                for prod in avoid_products:
                    with st.expander(f"❌ {prod['type']}: {prod['name']}"):
                        st.write(f"**Ingredients:** {prod['ingredients']}")
                        if st.button("Why is this not suitable?", key=f"btn_avoid_{prod['name']}"):
                            st.session_state['verify_request'] = {
                                'profile': features,
                                'product_name': prod['name'],
                                'ingredients': prod['ingredients']
                            }

    # Handling Groq Verification Display Outside the form
    if 'verify_request' in st.session_state:
        req = st.session_state['verify_request']
        st.markdown("---")
        st.subheader(f"🧠 AI Verification: {req['product_name']}")
        with st.spinner("Consulting Groq API for dermatologist-level analysis..."):
            analysis = groq_verifier.verify_product(
                req['profile'], 
                req['product_name'], 
                req['ingredients']
            )
            st.info(analysis)
            
    if 'user_profile' in st.session_state:
        st.markdown("---")
        st.subheader("🔍 Search & Verify Any Product")
        st.markdown("Want to check a specific product not listed above?")
        
        custom_product = st.text_input("Enter brand and product name (e.g., Cetaphil Gentle Cleanser)")
        if st.button("Verify Custom Product"):
            if custom_product.strip():
                with st.spinner("Asking AI Dermatologist..."):
                    analysis = groq_verifier.verify_product(
                        st.session_state['user_profile'], 
                        custom_product, 
                        ""
                    )
                    st.info(analysis)
            else:
                st.warning("Please enter a product name first.")

if __name__ == "__main__":
    run_app()
