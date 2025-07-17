import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

@st.cache_data
def load_and_prepare_model():
    """Load unprocessed dataset and train Random Forest model for prediction"""
    # Load unprocessed dataset
    df = pd.read_csv("datasets/Unprocessed_Obesity_Dataset.csv")
    
    # Calculate BMI
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # Get unique values for categorical features from the actual data
    categorical_values = {
        'Gender': df['Gender'].unique().tolist(),
        'FAVC': df['FAVC'].unique().tolist(),
        'CAEC': df['CAEC'].unique().tolist(),
        'SMOKE': df['SMOKE'].unique().tolist(),
        'SCC': df['SCC'].unique().tolist(),
        'CALC': df['CALC'].unique().tolist(),
        'family_history_with_overweight': df['family_history_with_overweight'].unique().tolist(),
        'MTRANS': df['MTRANS'].unique().tolist()
    }
    
    # Get unique target labels
    target_labels = df['NObeyesdad'].unique().tolist()
    
    # Create label encoders for categorical features
    label_encoders = {}
    categorical_cols = ['Gender', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 
                       'family_history_with_overweight', 'MTRANS', 'NObeyesdad']
    
    # Encode categorical features
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare features and target (excluding BMI to avoid data leakage)
    feature_cols = ['Age', 'Height', 'Weight', 'Gender', 'FAVC', 'FCVC', 
                   'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'family_history_with_overweight', 
                   'FAF', 'TUE', 'CALC', 'MTRANS']
    
    X = df_encoded[feature_cols]
    y = df_encoded['NObeyesdad']
    
    # Standardize numeric features (excluding BMI)
    numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    return model, scaler, label_encoders, categorical_values, target_labels, feature_cols

def predict_obesity(user_data, model, scaler, label_encoders, target_labels, feature_cols):
    """Make prediction using Random Forest model"""
    # Prepare user data
    user_df = pd.DataFrame([user_data])
    
    # Calculate BMI
    user_df['BMI'] = user_df['Weight'] / (user_df['Height'] ** 2)
    
    # Encode categorical features using label encoders
    categorical_cols = ['Gender', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 
                       'family_history_with_overweight', 'MTRANS']
    
    for col in categorical_cols:
        if col in user_df.columns and col in label_encoders:
            user_df[col] = label_encoders[col].transform(user_df[col])
    
    # Select only the features used in training (excluding BMI)
    user_df = user_df[feature_cols]
    
    # Scale numeric features (excluding BMI)
    numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])
    
    # Make prediction
    prediction = model.predict(user_df)[0]
    prediction_proba = model.predict_proba(user_df)[0]
    
    # Convert prediction back to original label using inverse transform
    predicted_label = label_encoders['NObeyesdad'].inverse_transform([prediction])[0]
    
    # Create probabilities dict with original labels
    original_labels = label_encoders['NObeyesdad'].inverse_transform(range(len(target_labels)))
    probabilities = dict(zip(original_labels, prediction_proba))
    
    return predicted_label, probabilities

def get_feature_importance(model, feature_columns):
    """Get feature importance from Random Forest model"""
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df

def main():
    st.set_page_config(page_title="Obesity Prediction", layout="centered")
    
    # Custom CSS to center content and control width
    st.markdown("""
    <style>
    .main > div {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title('🔮 Obesity Prediction Tool')
    st.markdown("Use Random Forest machine learning model to predict obesity level based on lifestyle and demographic factors")
    
    # Load model and preprocessors
    with st.spinner("Loading Random Forest model..."):
        model, scaler, label_encoders, categorical_values, target_labels, feature_columns = load_and_prepare_model()
    
    st.success("✅ Random Forest model loaded successfully!")
    
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
            fcvc = st.number_input("Frequency of consumption of vegetables (1-3)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            ncp = st.number_input("Number of main meals per day", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
            caec = st.selectbox("Consumption of food between meals", categorical_values.get('CAEC', ['no', 'Sometimes', 'Frequently', 'Always']))
            ch2o = st.number_input("Daily water consumption (L)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            
        with col3:
            st.subheader("Lifestyle & Health")
            faf = st.number_input("Physical activity frequency (0-3)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
            tue = st.number_input("Technology use time (0-2)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            calc = st.selectbox("Alcohol consumption", categorical_values.get('CALC', ['no', 'Sometimes', 'Frequently', 'Always']))
            smoke = st.selectbox("Do you smoke?", categorical_values.get('SMOKE', ['no', 'yes']))
            scc = st.selectbox("Calories consumption monitoring", categorical_values.get('SCC', ['no', 'yes']))
            family_history = st.selectbox("Family history with overweight", categorical_values.get('family_history_with_overweight', ['no', 'yes']))
            mtrans = st.selectbox("Transportation used", categorical_values.get('MTRANS', ['Walking', 'Bike', 'Public_Transportation', 'Motorbike', 'Automobile']))
        
        submitted = st.form_submit_button("🔮 Predict Obesity Level", type="primary")
        
        if submitted:
            # Prepare user data
            user_data = {
                'Age': age,
                'Gender': gender,
                'Height': height,
                'Weight': weight,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': ncp,
                'CAEC': caec,
                'SMOKE': smoke,
                'CH2O': ch2o,
                'SCC': scc,
                'family_history_with_overweight': family_history,
                'FAF': faf,
                'TUE': tue,
                'CALC': calc,
                'MTRANS': mtrans
            }
            
            # Calculate BMI
            bmi = weight / (height ** 2)
            st.info(f"📊 Your BMI: {bmi:.2f}")
            
            # Make prediction
            with st.spinner("Making prediction..."):
                prediction, probabilities = predict_obesity(user_data, model, scaler, label_encoders, target_labels, feature_columns)
            
            # Display results
            st.header('🎯 Prediction Results')
            
            # Main prediction result
            st.success(f"**Predicted Obesity Level: {prediction}**")
            
            # Probability breakdown
            st.subheader('📈 Prediction Probabilities')
            
            # Create probability dataframe
            prob_df = pd.DataFrame(list(probabilities.items()), columns=['Obesity Level', 'Probability'])
            prob_df = prob_df.sort_values('Probability', ascending=False)
            prob_df['Probability %'] = (prob_df['Probability'] * 100).round(1)
            
            # Display as chart
            import plotly.express as px
            
            fig = px.bar(prob_df, x='Obesity Level', y='Probability', 
                        title='Prediction Probabilities',
                        text='Probability %',
                        color='Probability',
                        color_continuous_scale='RdYlBu_r')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(
                height=400, 
                showlegend=False,
                margin=dict(t=60, b=40, l=40, r=40),
                yaxis=dict(range=[0, prob_df['Probability'].max() * 1.15])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader('🔍 What Influenced This Prediction?')
            
            # Get feature importance
            importance_df = get_feature_importance(model, feature_columns)
            
            # Display top 10 most important features
            top_features = importance_df.head(10)
            
            fig_importance = px.bar(
                top_features.iloc[::-1], 
                x='Importance', 
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Factors',
                text='Importance'
            )
            fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_importance.update_layout(
                height=400,
                margin=dict(t=60, b=40, l=120, r=60),
                xaxis=dict(range=[0, top_features['Importance'].max() * 1.15])
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Health recommendations
            st.subheader('💡 Personalized Health Recommendations')
            
            recommendations = []
            
            if bmi < 18.5:
                recommendations.append("🍎 Consider increasing caloric intake with nutritious foods")
                recommendations.append("💪 Include strength training to build healthy muscle mass")
            elif bmi >= 30:
                recommendations.append("🏃‍♀️ Consider increasing physical activity and consulting a healthcare professional")
                recommendations.append("🥗 Focus on a balanced, calorie-controlled diet")
            
            if faf < 2:
                recommendations.append("🏋️‍♂️ Try to increase physical activity to at least 150 minutes per week")
            
            if tue > 6:
                recommendations.append("📱 Consider reducing screen time to improve overall health")
            
            if ch2o < 2:
                recommendations.append("💧 Increase daily water consumption to at least 8 glasses per day")
            
            if fcvc < 2:
                recommendations.append("🥗 Include more vegetables in your daily diet (aim for 5+ servings)")
            
            if smoke == 'yes':
                recommendations.append("🚭 Consider quitting smoking for significant health improvements")
            
            if favc == 'yes':
                recommendations.append("🍔 Try to reduce consumption of high-caloric processed foods")
            
            if calc in ['Always', 'Frequently']:
                recommendations.append("🍷 Consider moderating alcohol consumption")
            
            if scc == 'no':
                recommendations.append("📱 Consider tracking your calorie intake to improve awareness")
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.write("✅ Keep up your excellent healthy lifestyle!")
    

    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 