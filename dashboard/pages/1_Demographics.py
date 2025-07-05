import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('datasets/Unprocessed_Obesity_Dataset.csv')
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 25, 35, 45, 55, 100], 
                            labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    return df

def main():
    st.set_page_config(page_title="Demographics Analysis", layout="wide")
    
    st.title('👥 Demographics Analysis')
    st.markdown("Deep dive into demographic patterns and obesity relationships")
    
    df = load_data()
    
    # Sidebar filters
    with st.sidebar:
        st.header('Filters')
        selected_states = st.multiselect('Select States', df['State'].unique(), default=df['State'].unique())
        age_range = st.slider('Age Range', int(df['Age'].min()), int(df['Age'].max()), 
                             (int(df['Age'].min()), int(df['Age'].max())))
        
    filtered_df = df[(df['State'].isin(selected_states)) & 
                     (df['Age'] >= age_range[0]) & 
                     (df['Age'] <= age_range[1])]
    
    # Section 1: Age Analysis
    st.header('📈 Age Distribution Analysis')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Age histogram by obesity category
        fig_age_hist = px.histogram(filtered_df, x='Age', color='NObeyesdad',
                                   title='Age Distribution by Obesity Category',
                                   nbins=20, barmode='overlay', opacity=0.7)
        st.plotly_chart(fig_age_hist, use_container_width=True)
    
    with col2:
        # Age group obesity rates
        age_obesity = filtered_df.groupby('Age_Group')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_age_obesity = px.bar(age_obesity, x='Age_Group', y='Obesity_Rate',
                                title='Obesity Rate by Age Group',
                                color='Obesity_Rate', color_continuous_scale='Reds')
        st.plotly_chart(fig_age_obesity, use_container_width=True)
    
    with col3:
        # Average BMI by age
        age_bmi = filtered_df.groupby('Age')['BMI'].mean().reset_index()
        fig_age_bmi = px.line(age_bmi, x='Age', y='BMI',
                             title='Average BMI by Age',
                             markers=True)
        st.plotly_chart(fig_age_bmi, use_container_width=True)
    
    # Section 2: Gender Analysis
    st.header('⚖️ Gender-Based Analysis')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution across obesity categories
        gender_obesity_cross = pd.crosstab(filtered_df['Gender'], filtered_df['NObeyesdad'], normalize='index') * 100
        fig_gender_cross = px.bar(gender_obesity_cross,
                                 title='Obesity Categories by Gender (%)',
                                 barmode='group')
        st.plotly_chart(fig_gender_cross, use_container_width=True)
    
    with col2:
        # BMI distribution by gender
        fig_bmi_gender = px.box(filtered_df, x='Gender', y='BMI', color='Gender',
                               title='BMI Distribution by Gender')
        st.plotly_chart(fig_bmi_gender, use_container_width=True)
    
    # Detailed age-gender analysis
    st.subheader('Age-Gender Interaction Analysis')
    
    # Create age-gender heatmap for obesity rates
    age_gender_obesity = filtered_df.groupby(['Age_Group', 'Gender']).apply(
        lambda x: (x['NObeyesdad'].str.contains('Obesity', na=False).sum() / len(x)) * 100
    ).reset_index(name='Obesity_Rate')
    
    age_gender_pivot = age_gender_obesity.pivot(index='Age_Group', columns='Gender', values='Obesity_Rate')
    
    fig_heatmap = px.imshow(age_gender_pivot.values,
                           x=age_gender_pivot.columns,
                           y=age_gender_pivot.index,
                           title='Obesity Rate Heatmap: Age Group vs Gender',
                           color_continuous_scale='Reds',
                           text_auto=True)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Section 3: Physical Characteristics
    st.header('📏 Physical Characteristics Analysis')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Height vs Weight scatter
        fig_height_weight = px.scatter(filtered_df, x='Height', y='Weight', 
                                      color='NObeyesdad', size='BMI',
                                      title='Height vs Weight by Obesity Category',
                                      hover_data=['Age', 'Gender'])
        st.plotly_chart(fig_height_weight, use_container_width=True)
    
    with col2:
        # BMI distribution
        fig_bmi_dist = px.histogram(filtered_df, x='BMI', color='Gender',
                                   title='BMI Distribution by Gender',
                                   nbins=30, opacity=0.7)
        st.plotly_chart(fig_bmi_dist, use_container_width=True)
    
    with col3:
        # Weight distribution by age group
        fig_weight_age = px.box(filtered_df, x='Age_Group', y='Weight', color='Age_Group',
                               title='Weight Distribution by Age Group')
        st.plotly_chart(fig_weight_age, use_container_width=True)
    
    # Section 4: Statistical Summary
    st.header('📊 Statistical Summary')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Demographics by Obesity Category')
        demo_summary = filtered_df.groupby('NObeyesdad').agg({
            'Age': ['mean', 'std'],
            'BMI': ['mean', 'std'],
            'Height': 'mean',
            'Weight': 'mean'
        }).round(2)
        st.dataframe(demo_summary)
    
    with col2:
        st.subheader('Gender Distribution by State')
        state_gender = pd.crosstab(filtered_df['State'], filtered_df['Gender'], margins=True)
        st.dataframe(state_gender)

if __name__ == "__main__":
    main()