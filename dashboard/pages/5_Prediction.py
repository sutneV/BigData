import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import os

@st.cache_data
def load_and_prepare_models():
    """Load dataset and train models for prediction"""
    # Load dataset
    df = pd.read_csv("datasets/Unprocessed_Obesity_Dataset.csv")
    if 'State' in df.columns:
        df = df.drop(columns=["State"])
    
    # Store original categorical values for user interface
    categorical_values = {}
    for column in df.select_dtypes(include='object').columns:
        if column != 'NObeyesdad':  # Don't include target variable
            categorical_values[column] = sorted(df[column].unique().tolist())
    
    # Store target variable labels
    target_labels = sorted(df['NObeyesdad'].unique().tolist())
    
    # Encode categorical features
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Standardize numeric features (excluding target)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols.remove('NObeyesdad')
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Prepare features and target
    X = df.drop(columns=['NObeyesdad'])
    y = df['NObeyesdad']
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                                n_estimators=100, max_depth=4, learning_rate=0.1)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
    
    return trained_models, scaler, label_encoders, categorical_values, target_labels, X.columns

def predict_obesity(user_data, models, scaler, label_encoders, target_labels):
    """Make predictions using all trained models"""
    predictions = {}
    probabilities = {}
    
    # Prepare user data
    user_df = pd.DataFrame([user_data])
    
    # Encode categorical features
    for column, le in label_encoders.items():
        if column in user_df.columns and column != 'NObeyesdad':
            user_df[column] = le.transform(user_df[column])
    
    # Scale numeric features
    numeric_cols = [col for col in user_df.columns if col in scaler.feature_names_in_]
    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])
    
    # Make predictions with each model
    for name, model in models.items():
        pred = model.predict(user_df)[0]
        pred_proba = model.predict_proba(user_df)[0]
        
        # Convert prediction back to original label
        predictions[name] = target_labels[pred]
        probabilities[name] = dict(zip(target_labels, pred_proba))
    
    return predictions, probabilities

def main():
    st.set_page_config(page_title="Obesity Prediction", layout="wide")
    
    st.title('🔮 Obesity Prediction Tool')
    st.markdown("Use machine learning models to predict obesity level based on lifestyle and demographic factors")
    
    # Load models and preprocessors
    with st.spinner("Loading models..."):
        models, scaler, label_encoders, categorical_values, target_labels, feature_columns = load_and_prepare_models()
    
    st.success("✅ Models loaded successfully!")
    
    # Create input form
    st.header('📝 Enter Your Information')
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", min_value=14, max_value=100, value=25, step=1)
            gender = st.selectbox("Gender", categorical_values.get('Gender', ['Female', 'Male']))
            height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            
        with col2:
            st.subheader("Eating Habits")
            favc = st.selectbox("Frequent consumption of high caloric food", categorical_values.get('FAVC', ['no', 'yes']))
            fcvc = st.number_input("Frequency of consumption of vegetables (0-3)", min_value=0, max_value=3, value=2, step=1)
            ncp = st.number_input("Number of main meals per day", min_value=1, max_value=5, value=3, step=1)
            caec = st.selectbox("Consumption of food between meals", categorical_values.get('CAEC', ['Always', 'Frequently', 'Sometimes', 'no']))
            ch2o = st.number_input("Daily water consumption (L)", min_value=1, max_value=4, value=2, step=1)
            
        with col3:
            st.subheader("Lifestyle & Health")
            faf = st.number_input("Physical activity frequency (days/week)", min_value=0, max_value=7, value=2, step=1)
            tue = st.number_input("Technology use time (hours/day)", min_value=0, max_value=12, value=2, step=1)
            calc = st.selectbox("Alcohol consumption", categorical_values.get('CALC', ['Always', 'Frequently', 'Sometimes', 'no']))
            smoke = st.selectbox("Do you smoke?", categorical_values.get('SMOKE', ['no', 'yes']))
            scc = st.selectbox("Calories consumption monitoring", categorical_values.get('SCC', ['no', 'yes']))
            family_history = st.selectbox("Family history with overweight", categorical_values.get('family_history_with_overweight', ['no', 'yes']))
            mtrans = st.selectbox("Transportation used", categorical_values.get('MTRANS', ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']))
        
        submitted = st.form_submit_button("🔮 Predict Obesity Level", type="primary")
        
        if submitted:
            # Prepare user data
            user_data = {
                'Age': age,
                'Gender': gender,
                'Height': height,
                'Weight': weight,
                'CALC': calc,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': ncp,
                'SCC': scc,
                'SMOKE': smoke,
                'CH2O': ch2o,
                'family_history_with_overweight': family_history,
                'FAF': faf,
                'TUE': tue,
                'CAEC': caec,
                'MTRANS': mtrans
            }
            
            # Calculate BMI
            bmi = weight / (height ** 2)
            st.info(f"📊 Your BMI: {bmi:.2f}")
            
            # Make predictions
            with st.spinner("Making predictions..."):
                predictions, probabilities = predict_obesity(user_data, models, scaler, label_encoders, target_labels)
            
            # Display results
            st.header('🎯 Prediction Results')
            
            # Model predictions comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Logistic Regression", predictions['Logistic Regression'])
                
            with col2:
                st.metric("Random Forest", predictions['Random Forest'])
                
            with col3:
                st.metric("XGBoost", predictions['XGBoost'])
            
            # Detailed probability breakdown
            st.subheader('📈 Prediction Probabilities')
            
            # Create probability dataframe for visualization
            prob_df = pd.DataFrame(probabilities).T
            prob_df = prob_df.round(3)
            
            # Display as a styled dataframe
            st.dataframe(prob_df, use_container_width=True)
            
            # Visualize probabilities
            import plotly.express as px
            
            # Reshape for plotting
            plot_data = []
            for model, probs in probabilities.items():
                for label, prob in probs.items():
                    plot_data.append({'Model': model, 'Obesity Level': label, 'Probability': prob})
            
            plot_df = pd.DataFrame(plot_data)
            
            fig = px.bar(plot_df, x='Obesity Level', y='Probability', color='Model',
                        title='Prediction Probabilities by Model',
                        barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Consensus prediction
            st.subheader('🤝 Consensus Prediction')
            
            # Find most common prediction
            pred_values = list(predictions.values())
            consensus = max(set(pred_values), key=pred_values.count)
            consensus_count = pred_values.count(consensus)
            
            if consensus_count >= 2:
                st.success(f"**Consensus Prediction: {consensus}**")
                st.write(f"✅ {consensus_count} out of 3 models agree on this prediction")
            else:
                st.warning("⚠️ Models disagree - consider consulting with healthcare professionals")
                st.write("Different models predict different obesity levels")
            
            # Health recommendations
            st.subheader('💡 Health Recommendations')
            
            recommendations = []
            
            if bmi < 18.5:
                recommendations.append("🍎 Consider increasing caloric intake with nutritious foods")
            elif bmi >= 30:
                recommendations.append("🏃‍♀️ Consider increasing physical activity and consulting a healthcare professional")
            
            if faf < 2:
                recommendations.append("🏋️‍♂️ Try to increase physical activity to at least 2-3 days per week")
            
            if tue > 6:
                recommendations.append("📱 Consider reducing screen time to improve overall health")
            
            if ch2o < 2:
                recommendations.append("💧 Increase daily water consumption for better hydration")
            
            if fcvc < 2:
                recommendations.append("🥗 Include more vegetables in your daily diet")
            
            if smoke == 'yes':
                recommendations.append("🚭 Consider quitting smoking for better health outcomes")
            
            if favc == 'yes':
                recommendations.append("🍔 Try to reduce consumption of high-caloric foods")
            
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.write("✅ Keep up your healthy lifestyle!")
    
    # Model information
    with st.expander("ℹ️ About the Models"):
        st.write("""
        **Model Information:**
        
        - **Logistic Regression**: A linear model that's interpretable and provides probability estimates
        - **Random Forest**: An ensemble method that combines multiple decision trees for robust predictions
        - **XGBoost**: A gradient boosting algorithm known for high performance on structured data
        
        **Features Used:**
        - Age, Gender, Height, Weight
        - Eating habits (caloric food consumption, vegetables, meals, snacking, water)
        - Lifestyle factors (physical activity, technology use, transportation)
        - Health behaviors (smoking, alcohol, calorie monitoring)
        - Family history of overweight
        
        **Obesity Categories:**
        - Insufficient_Weight
        - Normal_Weight  
        - Overweight_Level_I
        - Overweight_Level_II
        - Obesity_Type_I
        - Obesity_Type_II
        - Obesity_Type_III
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 