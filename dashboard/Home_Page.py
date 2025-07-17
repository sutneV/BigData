import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json

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
    
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                           labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    return df

@st.cache_data
def load_unprocessed_data():
    """Load unprocessed dataset for state mapping"""
    df_unprocessed = pd.read_csv('datasets/Unprocessed_Obesity_Dataset.csv')
    
    # Calculate BMI and obesity status
    df_unprocessed['BMI'] = df_unprocessed['Weight'] / (df_unprocessed['Height'] ** 2)
    df_unprocessed['Is_Obese'] = df_unprocessed['NObeyesdad'].apply(
        lambda x: 1 if 'Obesity' in str(x) else 0
    )
    df_unprocessed['Is_Overweight_Obese'] = df_unprocessed['NObeyesdad'].apply(
        lambda x: 1 if any(term in str(x) for term in ['Overweight', 'Obesity']) else 0
    )
    
    return df_unprocessed

@st.cache_data
def load_geojson():
    """Load Malaysia GeoJSON data"""
    with open('map_json/malaysia.geojson', 'r') as f:
        return json.load(f)

def create_state_mapping():
    """Create mapping between dataset and GeoJSON state names"""
    state_mapping = {
        'Wilayah Persekutuan Kuala Lumpur': 'W.P. Kuala Lumpur',
        'Wilayah Persekutuan Putrajaya': 'W.P. Putrajaya',
        'Pulau Pinang': 'Pulau Pinang',
        'Kedah': 'Kedah',
        'Perlis': 'Perlis',
        'Perak': 'Perak',
        'Selangor': 'Selangor',
        'Negeri Sembilan': 'Negeri Sembilan',
        'Melaka': 'Melaka',
        'Johor': 'Johor',
        'Pahang': 'Pahang',
        'Terengganu': 'Terengganu',
        'Kelantan': 'Kelantan',
        'Sabah': 'Sabah',
        'Sarawak': 'Sarawak'
    }
    return state_mapping

def calculate_state_metrics(df_unprocessed):
    """Calculate state-level metrics from unprocessed data"""
    state_metrics = df_unprocessed.groupby('State').agg({
        'Is_Obese': 'mean',
        'Is_Overweight_Obese': 'mean',
        'BMI': 'mean',
        'Age': 'mean',
        'Weight': 'mean',
        'Height': 'mean',
        'FAF': 'mean',
        'TUE': 'mean'
    }).round(2)
    
    state_metrics.columns = [
        'Obesity_Rate', 'Overweight_Obese_Rate', 'Avg_BMI', 'Avg_Age',
        'Avg_Weight', 'Avg_Height', 'Avg_Physical_Activity', 'Avg_Screen_Time'
    ]
    
    # Convert rates to percentages
    state_metrics['Obesity_Rate'] *= 100
    state_metrics['Overweight_Obese_Rate'] *= 100
    
    # Add sample size
    state_metrics['Sample_Size'] = df_unprocessed.groupby('State').size()
    
    # Map to GeoJSON state names
    state_mapping = create_state_mapping()
    state_metrics['GeoJSON_Name'] = state_metrics.index.map(state_mapping)
    
    return state_metrics.reset_index()



def main():
    st.set_page_config(page_title="Malaysian Obesity Analytics", layout="centered")
    
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
    st.title('🏠 Malaysian Obesity Analytics Dashboard')
    st.markdown("Comprehensive analysis of obesity patterns and health behaviors in Malaysia")
    
    # Load data
    df = load_and_prepare_data()
    
    # Key Statistics Overview
    st.header('📊 Key Statistics')
    
    col1, col2, col3 = st.columns(3)
    
    total_participants = len(df)
    obesity_rate = (df['Weight_Classification'].isin(['Obesity I', 'Obesity II', 'Obesity III']).sum() / total_participants) * 100
    overweight_rate = (df['Weight_Classification'] == 'Overweight').sum() / total_participants * 100
    normal_rate = (df['Weight_Classification'] == 'Normal').sum() / total_participants * 100
    
    # Additional variables for Key Findings section
    avg_bmi = df['BMI'].mean()
    overweight_obese_rate = (df['Weight_Classification'].isin(['Overweight', 'Obesity I', 'Obesity II', 'Obesity III']).sum() / total_participants) * 100
    
    with col1:
        st.metric("Obesity Rate", f"{obesity_rate:.1f}%")
    
    with col2:
        st.metric("Overweight Rate", f"{overweight_rate:.1f}%")
    
    with col3:
        st.metric("Normal Weight Rate", f"{normal_rate:.1f}%")
    
    # Overview Insights
    st.header('🔍 Overview Insights')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weight Classification Distribution
        weight_dist = df['Weight_Classification'].value_counts()
        fig_weight = px.pie(
            values=weight_dist.values, 
            names=weight_dist.index,
            title="Weight Classification Distribution",
            hole=0.3
        )
        fig_weight.update_layout(height=400)
        st.plotly_chart(fig_weight, use_container_width=True)
    
    with col2:
        # BMI by Age Group
        avg_bmi_age = df.groupby('Age_Group')['BMI'].mean().reset_index()
        fig_bmi_age = px.bar(
            avg_bmi_age, 
            x='Age_Group', 
            y='BMI',
            title="Average BMI by Age Group",
            text='BMI'
        )
        fig_bmi_age.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_bmi_age.update_layout(
            height=400,
            margin=dict(t=60, b=40, l=40, r=40),
            yaxis=dict(range=[0, avg_bmi_age['BMI'].max() * 1.15])
        )
        st.plotly_chart(fig_bmi_age, use_container_width=True)
    
    # Key Risk Factors Analysis
    st.header('⚠️ Key Risk Factors')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Family History Impact
        family_obesity = df.groupby('family_history_with_overweight')['Weight_Classification'].apply(
            lambda x: (x.isin(['Obesity I', 'Obesity II', 'Obesity III']).sum() / len(x)) * 100
        ).reset_index()
        family_obesity.columns = ['Family History', 'Obesity Rate']
        
        fig_family = px.bar(
            family_obesity,
            x='Family History',
            y='Obesity Rate',
            title='Obesity Rate by Family History',
            text='Obesity Rate'
        )
        fig_family.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_family.update_layout(
            height=400,
            margin=dict(t=60, b=40, l=40, r=40),
            yaxis=dict(range=[0, family_obesity['Obesity Rate'].max() * 1.15])
        )
        st.plotly_chart(fig_family, use_container_width=True)
    
    with col2:
        # Physical Activity Impact
        activity_obesity = df.groupby('FAF')['Weight_Classification'].apply(
            lambda x: (x.isin(['Obesity I', 'Obesity II', 'Obesity III']).sum() / len(x)) * 100
        ).reset_index()
        activity_obesity.columns = ['Physical Activity', 'Obesity Rate']
        
        fig_activity = px.bar(
            activity_obesity,
            x='Physical Activity',
            y='Obesity Rate',
            title='Obesity Rate by Physical Activity',
            text='Obesity Rate'
        )
        fig_activity.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_activity.update_layout(
            height=400,
            margin=dict(t=60, b=40, l=40, r=40),
            yaxis=dict(range=[0, activity_obesity['Obesity Rate'].max() * 1.15])
        )
        st.plotly_chart(fig_activity, use_container_width=True)
    
    with col3:
        # High Caloric Food Impact
        caloric_obesity = df.groupby('FAVC')['Weight_Classification'].apply(
            lambda x: (x.isin(['Obesity I', 'Obesity II', 'Obesity III']).sum() / len(x)) * 100
        ).reset_index()
        caloric_obesity.columns = ['High Caloric Food', 'Obesity Rate']
        
        fig_caloric = px.bar(
            caloric_obesity,
            x='High Caloric Food',
            y='Obesity Rate',
            title='Obesity Rate by High Caloric Food Consumption',
            text='Obesity Rate'
        )
        fig_caloric.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_caloric.update_layout(
            height=400,
            margin=dict(t=60, b=40, l=40, r=40),
            yaxis=dict(range=[0, caloric_obesity['Obesity Rate'].max() * 1.15])
        )
        st.plotly_chart(fig_caloric, use_container_width=True)
    
    # Interactive State Maps Section
    st.header('🌏 Interactive State Maps')
    
    # Load map data
    df_unprocessed = load_unprocessed_data()
    geojson = load_geojson()
    state_metrics = calculate_state_metrics(df_unprocessed)
    
    # Map metric selector
    col_selector, col_spacer = st.columns([1, 3])
    with col_selector:
        map_metric = st.selectbox(
            "Select Map Metric",
            options=['Obesity_Rate', 'Avg_BMI', 'Avg_Physical_Activity'],
            format_func=lambda x: {
                'Obesity_Rate': 'Obesity Rate (%)',
                'Avg_BMI': 'Average BMI',
                'Avg_Physical_Activity': 'Physical Activity Level'
            }[x]
        )
    
    # Main choropleth map (selected metric)
    color_scale = {
        'Obesity_Rate': 'reds',
        'Avg_BMI': 'blues',
        'Avg_Physical_Activity': 'greens'
    }
    
    # Calculate appropriate range for color scale
    metric_values = state_metrics[map_metric].dropna()
    color_range = [metric_values.min(), metric_values.quantile(0.98)]
    
    fig_map = px.choropleth_mapbox(
        state_metrics,
        geojson=geojson,
        locations='GeoJSON_Name',
        color=map_metric,
        color_continuous_scale=color_scale[map_metric],
        range_color=color_range,
        featureidkey="properties.name",
        mapbox_style="carto-positron",
        zoom=4.38,
        center={"lat": 4, "lon": 109.5},
        opacity=0.75,
        hover_data=['State', 'Sample_Size'],
        title=f"Malaysia: {map_metric.replace('_', ' ').title()}",
        labels={map_metric: map_metric.replace('_', ' ').title()}
    )
    
    # Standardize layout to ensure consistent sizing
    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=500,
        width=None,  # Let it use container width
        font=dict(size=12),
        coloraxis_colorbar=dict(
            thickness=15,
            len=0.7,
            x=1.02,
            xanchor="left"
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)
    


    # Navigation Section
    st.header('🧭 Explore Detailed Analysis')
    st.markdown("*Interactive maps are now included above. Navigate to other sections for detailed analysis:*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader('👥 Demographics')
        st.write('Analyze age, gender, and population patterns')
        if st.button('Go to Demographics', key='demo'):
            st.switch_page('pages/1_Demographics.py')
    
    with col2:
        st.subheader('🏃‍♂️ Lifestyle')
        st.write('Explore physical activity, diet, and habits')
        if st.button('Go to Lifestyle', key='lifestyle'):
            st.switch_page('pages/2_Lifestyle.py')
    
    with col3:
        st.subheader('🏥 Health Analysis')
        st.write('Examine health behaviors and risk factors')
        if st.button('Go to Health Analysis', key='health'):
            st.switch_page('pages/3_Health_Analysis.py')
    

    
    # Key Findings
    st.header('🔍 Key Findings')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Demographics Insights')
        st.write(f"• **{obesity_rate:.1f}%** of participants are classified as obese")
        st.write(f"• Average BMI across all participants is **{avg_bmi:.1f}**")
        st.write(f"• **{overweight_obese_rate:.1f}%** are overweight or obese")
        
        st.subheader('Lifestyle Patterns')
        never_exercise = (df['FAF'] == 'Never').sum() / len(df) * 100
        high_screen_time = (df['TUE'] == '>5 hours').sum() / len(df) * 100
        st.write(f"• **{never_exercise:.1f}%** never exercise")
        st.write(f"• **{high_screen_time:.1f}%** have >5 hours screen time daily")
    
    with col2:
        st.subheader('Health Risk Factors')
        family_history_rate = (df['family_history_with_overweight'] == 'Yes').sum() / len(df) * 100
        high_caloric_rate = (df['FAVC'] == 'Yes').sum() / len(df) * 100
        st.write(f"• **{family_history_rate:.1f}%** have family history of overweight")
        st.write(f"• **{high_caloric_rate:.1f}%** frequently consume high-caloric foods")
        
        st.subheader('Regional Variations')
        regional_obesity = df.groupby('State')['Weight_Classification'].apply(
            lambda x: (x.isin(['Obesity I', 'Obesity II', 'Obesity III']).sum() / len(x)) * 100
        )
        highest_region = regional_obesity.idxmax()
        lowest_region = regional_obesity.idxmin()
        st.write(f"• Highest obesity rate: **{highest_region}** ({regional_obesity.max():.1f}%)")
        st.write(f"• Lowest obesity rate: **{lowest_region}** ({regional_obesity.min():.1f}%)")

if __name__ == "__main__":
    main()
