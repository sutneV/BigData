import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv('datasets/Unprocessed_Obesity_Dataset.csv')
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # Create health risk score
    df['Health_Risk_Score'] = 0
    df.loc[df['family_history_with_overweight'] == 'yes', 'Health_Risk_Score'] += 2
    df.loc[df['FAVC'] == 'yes', 'Health_Risk_Score'] += 1
    df.loc[df['SMOKE'] == 'yes', 'Health_Risk_Score'] += 2
    df.loc[df['SCC'] == 'no', 'Health_Risk_Score'] += 1
    df.loc[df['CALC'] == 'Frequently', 'Health_Risk_Score'] += 1
    df.loc[df['FAF'] < 1, 'Health_Risk_Score'] += 1
    
    return df

def main():
    st.set_page_config(page_title="Health Behaviors", layout="wide")
    
    st.title('🏥 Health Behaviors Analysis')
    st.markdown("Comprehensive analysis of health behaviors and risk factors")
    
    df = load_data()
    
    # Sidebar risk assessment
    with st.sidebar:
        st.header('Personal Risk Assessment')
        
        user_age = st.number_input('Your Age', 18, 80, 25)
        user_family_history = st.selectbox('Family History of Overweight', ['yes', 'no'])
        user_smoking = st.selectbox('Smoking Habits', ['yes', 'no'])
        user_alcohol = st.selectbox('Alcohol Consumption', ['no', 'Sometimes', 'Frequently'])
        user_activity = st.number_input('Physical Activity (days/week)', 0, 7, 2)
        
        if st.button('Calculate My Risk'):
            risk_score = 0
            if user_family_history == 'yes': risk_score += 2
            if user_smoking == 'yes': risk_score += 2
            if user_alcohol == 'Frequently': risk_score += 1
            if user_activity < 1: risk_score += 1
            
            risk_level = ['Very Low', 'Low', 'Moderate', 'High', 'Very High'][min(risk_score, 4)]
            st.metric('Your Risk Level', risk_level)
            st.metric('Risk Score', f'{risk_score}/8')
    
    # Section 1: Smoking Analysis
    st.header('🚭 Smoking Habits Analysis')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Smoking distribution
        smoke_counts = df['SMOKE'].value_counts()
        fig_smoke = px.pie(values=smoke_counts.values, names=smoke_counts.index,
                          title='Smoking Habits Distribution',
                          color_discrete_sequence=['#90EE90', '#FF6B6B'])
        st.plotly_chart(fig_smoke, use_container_width=True)
    
    with col2:
        # Smoking vs obesity
        smoke_obesity = pd.crosstab(df['SMOKE'], df['NObeyesdad'], normalize='index') * 100
        fig_smoke_obesity = px.bar(smoke_obesity,
                                  title='Obesity Categories by Smoking Status (%)',
                                  barmode='group')
        st.plotly_chart(fig_smoke_obesity, use_container_width=True)
    
    with col3:
        # BMI by smoking status
        fig_bmi_smoke = px.box(df, x='SMOKE', y='BMI', color='SMOKE',
                              title='BMI Distribution by Smoking Status')
        st.plotly_chart(fig_bmi_smoke, use_container_width=True)
    
    # Section 2: Alcohol Consumption
    st.header('🍷 Alcohol Consumption Patterns')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Alcohol consumption distribution
        alcohol_counts = df['CALC'].value_counts()
        fig_alcohol = px.bar(x=alcohol_counts.index, y=alcohol_counts.values,
                            title='Alcohol Consumption Distribution',
                            color=alcohol_counts.values, color_continuous_scale='Blues')
        st.plotly_chart(fig_alcohol, use_container_width=True)
    
    with col2:
        # Alcohol vs obesity rate
        alcohol_obesity = df.groupby('CALC')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_alcohol_obesity = px.bar(alcohol_obesity, x='CALC', y='Obesity_Rate',
                                    title='Obesity Rate by Alcohol Consumption',
                                    color='Obesity_Rate', color_continuous_scale='Reds')
        st.plotly_chart(fig_alcohol_obesity, use_container_width=True)
    
    # Section 3: Eating Behaviors
    st.header('🍽️ Eating Behaviors')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # High caloric food consumption
        favc_obesity = pd.crosstab(df['FAVC'], df['NObeyesdad'], normalize='index') * 100
        fig_favc = px.bar(favc_obesity,
                         title='Obesity by High Caloric Food Consumption (%)',
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        st.plotly_chart(fig_favc, use_container_width=True)
    
    with col2:
        # Calorie monitoring
        scc_obesity = pd.crosstab(df['SCC'], df['NObeyesdad'], normalize='index') * 100
        fig_scc = px.bar(scc_obesity,
                        title='Obesity by Calorie Monitoring (%)',
                        color_discrete_sequence=['#FFB6C1', '#87CEEB'])
        st.plotly_chart(fig_scc, use_container_width=True)
    
    with col3:
        # Food between meals
        caec_counts = df['CAEC'].value_counts()
        fig_caec = px.pie(values=caec_counts.values, names=caec_counts.index,
                         title='Food Consumption Between Meals')
        st.plotly_chart(fig_caec, use_container_width=True)
    
    # Section 4: Family History Impact
    st.header('👨‍👩‍👧‍👦 Genetic and Family Factors')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Family history distribution
        family_counts = df['family_history_with_overweight'].value_counts()
        fig_family = px.pie(values=family_counts.values, names=family_counts.index,
                           title='Family History of Overweight Distribution',
                           color_discrete_sequence=['#98FB98', '#FFA07A'])
        st.plotly_chart(fig_family, use_container_width=True)
    
    with col2:
        # Family history impact on obesity
        family_obesity = pd.crosstab(df['family_history_with_overweight'], 
                                   df['NObeyesdad'].str.contains('Obesity', na=False),
                                   normalize='index') * 100
        
        fig_family_impact = px.bar(family_obesity,
                                  title='Obesity Rate by Family History',
                                  color_discrete_sequence=['#87CEEB', '#FF6B6B'])
        st.plotly_chart(fig_family_impact, use_container_width=True)
    
    # Section 5: Health Risk Assessment
    st.header('⚠️ Comprehensive Health Risk Assessment')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk score distribution
        risk_dist = df['Health_Risk_Score'].value_counts().sort_index()
        fig_risk_dist = px.bar(x=risk_dist.index, y=risk_dist.values,
                              title='Health Risk Score Distribution',
                              labels={'x': 'Risk Score', 'y': 'Count'},
                              color=risk_dist.values, color_continuous_scale='Reds')
        st.plotly_chart(fig_risk_dist, use_container_width=True)
    
    with col2:
        # Risk score vs obesity
        risk_obesity = df.groupby('Health_Risk_Score')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_risk_obesity = px.line(risk_obesity, x='Health_Risk_Score', y='Obesity_Rate',
                                  title='Obesity Rate by Health Risk Score',
                                  markers=True, line_shape='spline')
        st.plotly_chart(fig_risk_obesity, use_container_width=True)
    
    # Health recommendations based on behaviors
    st.header('🎯 Personalized Health Recommendations')
    
    high_risk_behaviors = []
    if df[df['SMOKE'] == 'yes'].shape[0] > 0:
        high_risk_behaviors.append('Smoking')
    if df[df['CALC'] == 'Frequently'].shape[0] > 0:
        high_risk_behaviors.append('Frequent Alcohol Consumption')
    if df[df['SCC'] == 'no'].shape[0] > 0:
        high_risk_behaviors.append('Not Monitoring Calories')
    
    recommendations = {
        'Smoking': [
            "Seek professional help for smoking cessation",
            "Use nicotine replacement therapy if recommended",
            "Join smoking cessation support groups",
            "Replace smoking with healthy stress management techniques"
        ],
        'Frequent Alcohol Consumption': [
            "Limit alcohol intake to recommended guidelines",
            "Choose alcohol-free days during the week",
            "Replace alcoholic drinks with water or low-calorie alternatives",
            "Seek support if you struggle with alcohol control"
        ],
        'Not Monitoring Calories': [
            "Start tracking daily caloric intake using apps",
            "Learn about portion sizes and calorie content",
            "Plan meals in advance",
            "Focus on nutrient-dense, lower-calorie foods"
        ]
    }
    
    for behavior in high_risk_behaviors:
        with st.expander(f"Recommendations for: {behavior}"):
            for i, rec in enumerate(recommendations[behavior], 1):
                st.write(f"{i}. {rec}")

if __name__ == "__main__":
    main()