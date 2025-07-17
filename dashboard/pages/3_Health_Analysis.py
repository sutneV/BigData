import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('datasets/Processed_Obesity_Dataset.csv')
    
    # Decode the numeric values back to meaningful labels
    # Gender mapping
    df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'})
    
    # Binary mappings
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map({0: 'No', 1: 'Yes'})
    df['FAVC'] = df['FAVC'].map({0: 'No', 1: 'Yes'})
    df['SMOKE'] = df['SMOKE'].map({0: 'No', 1: 'Yes'})
    df['SCC'] = df['SCC'].map({0: 'No', 1: 'Yes'})
    
    # Frequency mappings
    df['FCVC'] = df['FCVC'].map({0: 'Never', 1: 'Sometimes', 2: 'Always'})
    df['CAEC'] = df['CAEC'].map({0: 'No', 1: 'Sometimes', 2: 'Frequently', 3: 'Always'})
    df['CALC'] = df['CALC'].map({0: 'No', 1: 'Sometimes', 2: 'Frequently', 3: 'Always'})
    
    # Number of main meals
    df['NCP'] = df['NCP'].map({1: '1-2 meals', 2: '3 meals', 3: '3+ meals'})
    
    # Water consumption
    df['CH2O'] = df['CH2O'].map({1: '<1L', 2: '1-2L', 3: '>2L'})
    
    # Physical activity frequency
    df['FAF'] = df['FAF'].map({0: 'Never', 1: '1-2x/week', 2: '2-3x/week', 3: '4-5x/week'})
    
    # Technology use
    df['TUE'] = df['TUE'].map({1: '0-2 hours', 2: '3-5 hours', 3: '>5 hours'})
    
    # Transportation
    df['MTRANS'] = df['MTRANS'].map({1: 'Walking', 2: 'Bike', 3: 'Public Transport', 4: 'Motorbike', 5: 'Car'})
    
    # State regions
    df['State'] = df['State'].map({1: 'Central', 2: 'Southern', 3: 'Northern', 4: 'East Coast', 5: 'East Malaysia'})
    
    # Weight classification
    df['Weight_Classification'] = df['Weight_Classification'].map({
        1: 'Underweight', 2: 'Normal', 3: 'Overweight', 
        4: 'Obesity I', 5: 'Obesity II', 6: 'Obesity III'
    })
    
    # Create age groups for analysis
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 25, 35, 45, 55, 100], 
                            labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Create health risk score
    df['Health_Risk_Score'] = 0
    df.loc[df['family_history_with_overweight'] == 'Yes', 'Health_Risk_Score'] += 2
    df.loc[df['FAVC'] == 'Yes', 'Health_Risk_Score'] += 1
    df.loc[df['SMOKE'] == 'Yes', 'Health_Risk_Score'] += 2
    df.loc[df['SCC'] == 'No', 'Health_Risk_Score'] += 1
    df.loc[df['CALC'].isin(['Frequently', 'Always']), 'Health_Risk_Score'] += 1
    df.loc[df['FAF'] == 'Never', 'Health_Risk_Score'] += 1
    
    # Categorize risk levels
    df['Risk_Level'] = pd.cut(df['Health_Risk_Score'], 
                             bins=[-1, 1, 3, 5, 8], 
                             labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'])
    
    return df

def main():
    st.set_page_config(page_title="Health Analysis", layout="centered")
    
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
    st.title('🏥 Health Analysis')
    st.markdown("Comprehensive analysis of health behaviors, risk factors, and obesity patterns")
    
    # Load data
    df = load_and_prepare_data()
    
    # Sidebar - Personal Risk Assessment
    with st.sidebar:
        st.header('🔍 Personal Risk Assessment')
        
        st.markdown("**Calculate your obesity risk based on health behaviors:**")
        
        user_age = st.number_input('Your Age', 18, 80, 25)
        user_family_history = st.selectbox('Family History of Overweight', ['No', 'Yes'])
        user_smoking = st.selectbox('Smoking Habits', ['No', 'Yes'])
        user_alcohol = st.selectbox('Alcohol Consumption', ['No', 'Sometimes', 'Frequently', 'Always'])
        user_activity = st.selectbox('Physical Activity', ['Never', '1-2x/week', '2-3x/week', '4-5x/week'])
        user_high_caloric = st.selectbox('High Caloric Food Consumption', ['No', 'Yes'])
        user_calorie_monitoring = st.selectbox('Monitor Calorie Consumption', ['No', 'Yes'])
        
        if st.button('🧮 Calculate My Risk', type="primary"):
            risk_score = 0
            if user_family_history == 'Yes': risk_score += 2
            if user_smoking == 'Yes': risk_score += 2
            if user_high_caloric == 'Yes': risk_score += 1
            if user_calorie_monitoring == 'No': risk_score += 1
            if user_alcohol in ['Frequently', 'Always']: risk_score += 1
            if user_activity == 'Never': risk_score += 1
            
            if risk_score <= 1:
                risk_level = 'Low Risk'
                risk_color = '#4CAF50'
            elif risk_score <= 3:
                risk_level = 'Moderate Risk'
                risk_color = '#FF9800'
            elif risk_score <= 5:
                risk_level = 'High Risk'
                risk_color = '#FF5722'
            else:
                risk_level = 'Very High Risk'
                risk_color = '#F44336'
            
            st.markdown(f"""
            <div style="background: {risk_color}; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; margin: 1rem 0;">
            <h3>Your Risk Level</h3>
            <h2>{risk_level}</h2>
            <p>Risk Score: {risk_score}/8</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Health filters
        st.markdown("---")
        st.header('🔍 Data Filters')
        
        # Smoking filter
        smoking_options = st.multiselect(
            'Smoking Habits',
            df['SMOKE'].unique(),
            default=df['SMOKE'].unique()
        )
        
        # Alcohol consumption filter
        alcohol_options = st.multiselect(
            'Alcohol Consumption',
            df['CALC'].unique(),
            default=df['CALC'].unique()
        )
        
        # Family history filter
        family_history_options = st.multiselect(
            'Family History',
            df['family_history_with_overweight'].unique(),
            default=df['family_history_with_overweight'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['SMOKE'].isin(smoking_options)) & 
        (df['CALC'].isin(alcohol_options)) &
        (df['family_history_with_overweight'].isin(family_history_options))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    # Define obesity categories for analysis
    obesity_categories = ['Obesity I', 'Obesity II', 'Obesity III']
    
    # Health Risk Factors Analysis
    st.header('⚠️ Health Risk Factors')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Family history impact
        family_obesity = filtered_df.groupby('family_history_with_overweight')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_family = px.bar(
            family_obesity, x='family_history_with_overweight', y='Obesity_Rate',
            title='Obesity Rate by Family History',
            color='Obesity_Rate',
            color_continuous_scale='Reds',
            labels={'family_history_with_overweight': 'Family History of Overweight', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_family.update_layout(height=400)
        st.plotly_chart(fig_family, use_container_width=True)
    
    with col2:
        # Risk level distribution
        risk_counts = filtered_df['Risk_Level'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            title='Health Risk Level Distribution',
            color_discrete_sequence=['#4CAF50', '#FF9800', '#FF5722', '#F44336']
        )
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Health Behaviors Analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Smoking vs obesity
        smoking_obesity = filtered_df.groupby('SMOKE')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_smoking = px.bar(
            smoking_obesity, x='SMOKE', y='Obesity_Rate',
            title='Obesity Rate by Smoking Habits',
            color='Obesity_Rate',
            color_continuous_scale='Reds',
            labels={'SMOKE': 'Smoking Habits', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_smoking.update_layout(height=350)
        st.plotly_chart(fig_smoking, use_container_width=True)
    
    with col2:
        # Alcohol consumption vs obesity
        alcohol_obesity = filtered_df.groupby('CALC')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_alcohol = px.bar(
            alcohol_obesity, x='CALC', y='Obesity_Rate',
            title='Obesity Rate by Alcohol Consumption',
            color='Obesity_Rate',
            color_continuous_scale='Oranges',
            labels={'CALC': 'Alcohol Consumption', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_alcohol.update_layout(height=350)
        st.plotly_chart(fig_alcohol, use_container_width=True)
    
    with col3:
        # Calorie monitoring vs obesity
        scc_obesity = filtered_df.groupby('SCC')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_scc = px.bar(
            scc_obesity, x='SCC', y='Obesity_Rate',
            title='Obesity Rate by Calorie Monitoring',
            color='Obesity_Rate',
            color_continuous_scale='Blues',
            labels={'SCC': 'Monitors Calorie Consumption', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_scc.update_layout(height=350)
        st.plotly_chart(fig_scc, use_container_width=True)
    
    st.markdown("---")
    
    # Risk Score Analysis
    st.header('📈 Risk Score Analysis')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk score vs obesity rate
        risk_score_obesity = filtered_df.groupby('Health_Risk_Score')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_risk_score = px.bar(
            risk_score_obesity, x='Health_Risk_Score', y='Obesity_Rate',
            title='Obesity Rate by Health Risk Score',
            color='Obesity_Rate',
            color_continuous_scale='Reds',
            labels={'Health_Risk_Score': 'Health Risk Score (0-8)', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_risk_score.update_layout(height=400)
        st.plotly_chart(fig_risk_score, use_container_width=True)
    
    with col2:
        # Average BMI by risk level
        bmi_risk = filtered_df.groupby('Risk_Level')['BMI'].mean().reset_index()
        
        fig_bmi_risk = px.bar(
            bmi_risk, x='Risk_Level', y='BMI',
            title='Average BMI by Risk Level',
            color='BMI',
            color_continuous_scale='Reds',
            labels={'Risk_Level': 'Risk Level', 'BMI': 'Average BMI'}
        )
        fig_bmi_risk.update_layout(height=400)
        st.plotly_chart(fig_bmi_risk, use_container_width=True)
    
    # Health Insights and Recommendations
    st.header('💡 Health Insights & Recommendations')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="risk-card">
        <h4>⚠️ Key Risk Factors Identified</h4>
        <ul>
        <li><strong>Family History:</strong> Strong genetic predisposition to obesity</li>
        <li><strong>Smoking:</strong> Associated with poor lifestyle choices and metabolism</li>
        <li><strong>Alcohol Consumption:</strong> Regular drinking correlates with weight gain</li>
        <li><strong>Poor Monitoring:</strong> Not tracking calories leads to overconsumption</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="recommendation-card">
        <h4>🎯 Health Recommendations</h4>
        <ul>
        <li><strong>Regular Health Checkups:</strong> Monitor BMI and health indicators</li>
        <li><strong>Smoking Cessation:</strong> Quit smoking for overall health improvement</li>
        <li><strong>Moderate Alcohol:</strong> Limit alcohol consumption</li>
        <li><strong>Calorie Awareness:</strong> Track daily caloric intake</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Health Analysis | Malaysian Obesity Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()