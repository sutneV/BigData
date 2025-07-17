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
    
    return df

def main():
    st.set_page_config(page_title="Lifestyle Analysis", layout="centered")
    
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
    st.title('🏃‍♂️ Lifestyle Analysis')
    st.markdown("Comprehensive analysis of physical activity, eating habits, and lifestyle behaviors")
    
    # Load data
    df = load_and_prepare_data()
    
    # Sidebar filters
    with st.sidebar:
        st.header('🔍 Lifestyle Filters')
        
        # Physical activity filter
        activity_levels = st.multiselect(
            'Physical Activity Frequency',
            df['FAF'].unique(),
            default=df['FAF'].unique(),
            help="Filter by physical activity frequency"
        )
        
        # Technology use filter
        tech_levels = st.multiselect(
            'Technology Use Time',
            df['TUE'].unique(),
            default=df['TUE'].unique(),
            help="Filter by daily technology use"
        )
        
        # Transportation filter
        transport_types = st.multiselect(
            'Transportation Method',
            df['MTRANS'].unique(),
            default=df['MTRANS'].unique()
        )
        
        # High caloric food consumption
        favc_options = st.multiselect(
            'High Caloric Food Consumption',
            df['FAVC'].unique(),
            default=df['FAVC'].unique()
        )
        
        # Age group filter
        age_groups = st.multiselect(
            'Age Groups',
            df['Age_Group'].unique(),
            default=df['Age_Group'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['FAF'].isin(activity_levels)) & 
        (df['TUE'].isin(tech_levels)) &
        (df['MTRANS'].isin(transport_types)) &
        (df['FAVC'].isin(favc_options)) &
        (df['Age_Group'].isin(age_groups))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    # Key Lifestyle Metrics
    st.header('📊 Lifestyle Overview')
    
    obesity_categories = ['Obesity I', 'Obesity II', 'Obesity III']
    total_participants = len(filtered_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label='👥 Total Participants', value=f"{total_participants:,}")
    
    with col2:
        active_people = filtered_df[filtered_df['FAF'].isin(['2-3x/week', '4-5x/week'])].shape[0]
        active_percentage = (active_people / total_participants) * 100
        st.metric(label='🏃‍♂️ Active People', value=f"{active_percentage:.1f}%")
    
    with col3:
        high_caloric = filtered_df[filtered_df['FAVC'] == 'Yes'].shape[0]
        high_caloric_percentage = (high_caloric / total_participants) * 100
        st.metric(label='🍔 High Caloric Food', value=f"{high_caloric_percentage:.1f}%")
    
    with col4:
        high_tech_use = filtered_df[filtered_df['TUE'] == '>5 hours'].shape[0]
        high_tech_percentage = (high_tech_use / total_participants) * 100
        st.metric(label='📱 High Tech Use', value=f"{high_tech_percentage:.1f}%")
    
    with col5:
        car_users = filtered_df[filtered_df['MTRANS'] == 'Car'].shape[0]
        car_percentage = (car_users / total_participants) * 100
        st.metric(label='🚗 Car Users', value=f"{car_percentage:.1f}%")
    
    st.markdown("---")
    
    # Physical Activity Analysis
    st.header('💪 Physical Activity Patterns')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Physical activity distribution
        activity_counts = filtered_df['FAF'].value_counts()
        fig_activity = px.pie(
            values=activity_counts.values, 
            names=activity_counts.index,
            title='Physical Activity Frequency Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_activity.update_layout(height=400)
        st.plotly_chart(fig_activity, use_container_width=True)
    
    with col2:
        # Obesity rate by activity level
        activity_obesity = filtered_df.groupby('FAF')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_activity_obesity = px.bar(
            activity_obesity, x='FAF', y='Obesity_Rate',
            title='Obesity Rate by Physical Activity Frequency',
            color='Obesity_Rate',
            color_continuous_scale='RdYlGn_r',
            labels={'FAF': 'Physical Activity Frequency', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_activity_obesity.update_layout(height=400)
        st.plotly_chart(fig_activity_obesity, use_container_width=True)
    
    st.markdown("---")
    
    # Eating Habits Analysis
    st.header('🍽️ Eating Habits Analysis')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # High caloric food consumption impact
        favc_obesity = filtered_df.groupby('FAVC')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_favc = px.bar(
            favc_obesity, x='FAVC', y='Obesity_Rate',
            title='Obesity Rate by High Caloric Food Consumption',
            color='Obesity_Rate',
            color_continuous_scale='Oranges',
            labels={'FAVC': 'High Caloric Food', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_favc.update_layout(height=350)
        st.plotly_chart(fig_favc, use_container_width=True)
    
    with col2:
        # Vegetable consumption analysis
        fcvc_obesity = filtered_df.groupby('FCVC')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_fcvc = px.bar(
            fcvc_obesity, x='FCVC', y='Obesity_Rate',
            title='Obesity Rate by Vegetable Consumption',
            color='Obesity_Rate',
            color_continuous_scale='Greens',
            labels={'FCVC': 'Vegetable Consumption', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_fcvc.update_layout(height=350)
        st.plotly_chart(fig_fcvc, use_container_width=True)
    
    with col3:
        # Number of main meals
        ncp_obesity = filtered_df.groupby('NCP')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_ncp = px.bar(
            ncp_obesity, x='NCP', y='Obesity_Rate',
            title='Obesity Rate by Number of Main Meals',
            color='Obesity_Rate',
            color_continuous_scale='Blues',
            labels={'NCP': 'Number of Main Meals', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_ncp.update_layout(height=350)
        st.plotly_chart(fig_ncp, use_container_width=True)
    
    # Additional eating habits
    col1, col2 = st.columns(2)
    
    with col1:
        # Food between meals
        caec_counts = filtered_df['CAEC'].value_counts()
        fig_caec = px.bar(
            x=caec_counts.index, y=caec_counts.values,
            title='Food Consumption Between Meals',
            labels={'x': 'Frequency', 'y': 'Count'},
            color=caec_counts.values,
            color_continuous_scale='Oranges'
        )
        fig_caec.update_layout(height=350)
        st.plotly_chart(fig_caec, use_container_width=True)
    
    with col2:
        # Water consumption analysis
        ch2o_obesity = filtered_df.groupby('CH2O')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_ch2o = px.bar(
            ch2o_obesity, x='CH2O', y='Obesity_Rate',
            title='Obesity Rate by Water Consumption',
            color='Obesity_Rate',
            color_continuous_scale='Blues',
            labels={'CH2O': 'Daily Water Consumption', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_ch2o.update_layout(height=350)
        st.plotly_chart(fig_ch2o, use_container_width=True)
    
    st.markdown("---")
    
    # Technology Use and Transportation
    st.header('📱 Technology Use & Transportation')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Technology use vs obesity
        tue_obesity = filtered_df.groupby('TUE')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_tue = px.bar(
            tue_obesity, x='TUE', y='Obesity_Rate',
            title='Obesity Rate by Technology Use Time',
            color='Obesity_Rate',
            color_continuous_scale='Reds',
            labels={'TUE': 'Daily Technology Use', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_tue.update_layout(height=400)
        st.plotly_chart(fig_tue, use_container_width=True)
    
    with col2:
        # Transportation method vs obesity
        transport_obesity = filtered_df.groupby('MTRANS')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_transport = px.bar(
            transport_obesity, x='Obesity_Rate', y='MTRANS',
            orientation='h',
            title='Obesity Rate by Transportation Method',
            color='Obesity_Rate',
            color_continuous_scale='Viridis',
            labels={'MTRANS': 'Transportation Method', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_transport.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_transport, use_container_width=True)
    

    
    # Key Lifestyle Insights
    st.header('💡 Key Lifestyle Insights')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card">
        <h4>💪 Physical Activity Impact</h4>
        <ul>
        <li><strong>Active Lifestyle:</strong> Regular exercise (2-3x/week or more) significantly reduces obesity risk</li>
        <li><strong>Sedentary Risk:</strong> People who never exercise show highest obesity rates</li>
        <li><strong>Gender Differences:</strong> Activity benefits are consistent across genders</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
        <h4>🍽️ Eating Habits Impact</h4>
        <ul>
        <li><strong>High Caloric Foods:</strong> Regular consumption strongly correlates with obesity</li>
        <li><strong>Vegetable Intake:</strong> Higher vegetable consumption associated with lower obesity rates</li>
        <li><strong>Meal Patterns:</strong> Number of main meals affects weight management</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Lifestyle Analysis | Malaysian Obesity Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()