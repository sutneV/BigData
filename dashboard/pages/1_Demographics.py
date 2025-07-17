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
    # Page configuration
    st.set_page_config(page_title="Demographics Analysis", layout="centered", initial_sidebar_state="expanded")
    
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
    st.title('👥 Demographics Analysis')
    st.markdown("Comprehensive analysis of age, gender, and population characteristics in obesity patterns")
    
    # Load data
    df = load_and_prepare_data()
    
    # Sidebar filters
    with st.sidebar:
        st.header('🔍 Data Filters')
        
        # Region filter
        all_regions = df['State'].unique()
        selected_regions = st.multiselect(
            'Select Regions', 
            all_regions, 
            default=all_regions,
            help="Filter data by Malaysian regions"
        )
        
        # Age range filter
        age_range = st.slider(
            'Age Range', 
            int(df['Age'].min()), 
            int(df['Age'].max()), 
            (int(df['Age'].min()), int(df['Age'].max())),
            help="Select age range for analysis"
        )
        
        # Gender filter
        genders = st.multiselect(
            'Select Gender',
            df['Gender'].unique(),
            default=df['Gender'].unique()
        )
        
        # Weight classification filter
        weight_classes = st.multiselect(
            'Weight Classifications',
            df['Weight_Classification'].unique(),
            default=df['Weight_Classification'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['State'].isin(selected_regions)) & 
        (df['Age'] >= age_range[0]) & 
        (df['Age'] <= age_range[1]) &
        (df['Gender'].isin(genders)) &
        (df['Weight_Classification'].isin(weight_classes))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    # Key Demographics Metrics
    st.header('📊 Key Demographics Overview')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_participants = len(filtered_df)
        st.metric(label='👥 Total Participants', value=f"{total_participants:,}")
    
    with col2:
        avg_age = filtered_df['Age'].mean()
        st.metric(label='📅 Average Age', value=f"{avg_age:.1f} years")
    
    with col3:
        male_percentage = (filtered_df[filtered_df['Gender'] == 'Male'].shape[0] / total_participants) * 100
        st.metric(label='👨 Male Participants', value=f"{male_percentage:.1f}%")
    
    with col4:
        female_percentage = (filtered_df[filtered_df['Gender'] == 'Female'].shape[0] / total_participants) * 100
        st.metric(label='👩 Female Participants', value=f"{female_percentage:.1f}%")
    
    with col5:
        avg_bmi = filtered_df['BMI'].mean()
        st.metric(label='📈 Average BMI', value=f"{avg_bmi:.1f}")
    
    st.markdown("---")
    
    # Age Analysis
    st.header('📈 Age Distribution Analysis')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by gender
        fig_age_gender = px.histogram(
            filtered_df, x='Age', color='Gender',
            title='Age Distribution by Gender',
            nbins=20, opacity=0.7,
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        fig_age_gender.update_layout(height=400)
        st.plotly_chart(fig_age_gender, use_container_width=True)
    
    with col2:
        # Age group obesity rates
        obesity_categories = ['Obesity I', 'Obesity II', 'Obesity III']
        age_obesity = filtered_df.groupby('Age_Group')['Weight_Classification'].apply(
            lambda x: (x.isin(obesity_categories).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_age_obesity = px.bar(
            age_obesity, x='Age_Group', y='Obesity_Rate',
            title='Obesity Rate by Age Group',
            color='Obesity_Rate', 
            color_continuous_scale='Reds',
            labels={'Age_Group': 'Age Group', 'Obesity_Rate': 'Obesity Rate (%)'}
        )
        fig_age_obesity.update_layout(height=400)
        st.plotly_chart(fig_age_obesity, use_container_width=True)
    
    # BMI Analysis by Age
    col1, col2 = st.columns(2)
    
    with col1:
        # BMI distribution by age group
        fig_bmi_age = px.box(
            filtered_df, x='Age_Group', y='BMI', 
            color='Age_Group',
            title='BMI Distribution by Age Group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bmi_age.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bmi_age, use_container_width=True)
    
    with col2:
        # Average BMI trend by age
        age_bmi_trend = filtered_df.groupby('Age')['BMI'].mean().reset_index()
        fig_bmi_trend = px.line(
            age_bmi_trend, x='Age', y='BMI',
            title='BMI Trend by Age',
            markers=True,
            color_discrete_sequence=['#667eea']
        )
        fig_bmi_trend.update_layout(height=400)
        st.plotly_chart(fig_bmi_trend, use_container_width=True)
    
    st.markdown("---")
    
    # Gender Analysis
    st.header('⚖️ Gender-Based Analysis')
    
    # Gender distribution across weight classifications
    gender_weight_cross = pd.crosstab(
        filtered_df['Gender'], 
        filtered_df['Weight_Classification'], 
        normalize='index'
    ) * 100
    
    fig_gender_weight = px.bar(
        gender_weight_cross,
        title='Weight Classification Distribution by Gender (%)',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_gender_weight.update_layout(height=500)
    st.plotly_chart(fig_gender_weight, use_container_width=True)
    
    # Age-Gender Interaction Analysis
    st.subheader('👥 Age-Gender Interaction')
    
    # Average BMI by age group and gender
    bmi_age_gender = filtered_df.groupby(['Age_Group', 'Gender'])['BMI'].mean().reset_index()
    
    fig_bmi_age_gender = px.bar(
        bmi_age_gender, x='Age_Group', y='BMI', 
        color='Gender',
        title='Average BMI by Age Group and Gender',
        barmode='group',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    fig_bmi_age_gender.update_layout(height=500)
    st.plotly_chart(fig_bmi_age_gender, use_container_width=True)
    
    st.markdown("---")
    
    # Physical Characteristics Analysis
    st.header('📏 Physical Characteristics')
    
    # Height vs Weight scatter plot
    fig_scatter = px.scatter(
        filtered_df, x='Height', y='Weight', 
        color='Gender',
        title='Scatter Plot of Height vs Weight',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
        opacity=0.7
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Key Insights
    st.header('💡 Key Demographic Insights')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
        <h4>🎯 Age Patterns</h4>
        <ul>
        <li><strong>Peak Age Range:</strong> Most participants are in the 26-35 age group</li>
        <li><strong>BMI Trend:</strong> BMI tends to increase with age up to middle age</li>
        <li><strong>Obesity Risk:</strong> Higher obesity rates observed in older age groups</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        gender_insights = filtered_df['Gender'].value_counts()
        male_pct = (gender_insights.get('Male', 0) / len(filtered_df)) * 100
        female_pct = (gender_insights.get('Female', 0) / len(filtered_df)) * 100
        
        st.markdown(f"""
        <div class="section-card">
        <h4>⚖️ Gender Distribution</h4>
        <ul>
        <li><strong>Male Participants:</strong> {male_pct:.1f}%</li>
        <li><strong>Female Participants:</strong> {female_pct:.1f}%</li>
        <li><strong>Gender Balance:</strong> {"Relatively balanced" if abs(male_pct - female_pct) < 10 else "Imbalanced distribution"}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Demographics Analysis | Malaysian Obesity Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()