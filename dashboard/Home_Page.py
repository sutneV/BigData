import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('datasets/Unprocessed_Obesity_Dataset.csv')
    
    # Calculate BMI
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 25, 35, 45, 55, 100], 
                            labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Create BMI categories
    df['BMI_Category'] = pd.cut(df['BMI'], 
                               bins=[0, 18.5, 25, 30, 35, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese Class I', 'Obese Class II+'])
    
    return df

@st.cache_data
def load_geojson():
    with open('map_json/malaysia.geojson') as f:
        return json.load(f)

def main():
    # Page configuration
    st.set_page_config(page_title="Malaysian Obesity Analytics", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS styling
    st.markdown("""
    <style>
    div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
        height: 5rem;
        width: auto;
    }   
    div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
    div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
        display: flex;
        align-items: center;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.write("🧭 **Navigate this page:**")
        st.markdown('''
        - [Welcome](#welcome-to-malaysian-obesity-analytics)
        - [Key Statistics](#obesity-at-a-glance)
        - [Demographics Overview](#demographics-overview)
        - [Obesity Distribution](#obesity-category-distribution)
        - [Geographic Analysis](#geographic-analysis)
        - [Age & Gender Analysis](#age-and-gender-analysis)
        - [Physical Measurements](#physical-measurements-analysis)
        - [Lifestyle Factors](#lifestyle-factors-analysis)
        - [Eating Habits](#eating-habits-analysis)
        - [Physical Activity](#physical-activity-analysis)
        - [Transportation Patterns](#transportation-patterns)
        - [Health Behaviors](#health-behaviors-analysis)
        - [Risk Factors](#risk-factors-assessment)
        - [Correlations](#correlation-analysis)
        - [Advanced Analytics](#advanced-analytics)
        - [Select Topics](#select-a-topic-to-explore)
        ''')
    
    # Load data
    df = load_and_prepare_data()
    geojson = load_geojson()
    
    # Title and description
    st.title('🏥 Welcome to Malaysian Obesity Analytics Dashboard')
    st.markdown("""
    <div style="font-size: 1.2em; text-align: justify; margin-bottom: 2rem;">
    The Malaysian Obesity Analytics Dashboard provides comprehensive insights into obesity patterns, 
    lifestyle factors, and health behaviors across Malaysia. This platform analyzes demographic data, 
    eating habits, physical activity levels, and geographic distributions to understand the obesity 
    landscape and support evidence-based health interventions.
    </div>
    """, unsafe_allow_html=True)
    
    # Section 1: Key Statistics at a Glance
    st.header('📊 Obesity at a Glance')
    st.caption('Comprehensive overview of obesity statistics in Malaysia')
    
    # Calculate key metrics
    total_participants = len(df)
    obesity_count = df[df['NObeyesdad'].str.contains('Obesity', na=False)].shape[0]
    overweight_count = df[df['NObeyesdad'].str.contains('Overweight', na=False)].shape[0]
    normal_weight_count = df[df['NObeyesdad'] == 'Normal_Weight'].shape[0]
    insufficient_weight_count = df[df['NObeyesdad'] == 'Insufficient_Weight'].shape[0]
    
    obesity_rate = (obesity_count / total_participants) * 100
    overweight_rate = (overweight_count / total_participants) * 100
    normal_rate = (normal_weight_count / total_participants) * 100
    avg_age = df['Age'].mean()
    avg_bmi = df['BMI'].mean()
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label='👥 Total Participants', value=f"{total_participants:,}")
    with col2:
        st.metric(label='🔴 Obesity Rate', value=f"{obesity_rate:.1f}%")
    with col3:
        st.metric(label='🟡 Overweight Rate', value=f"{overweight_rate:.1f}%")
    with col4:
        st.metric(label='🟢 Normal Weight Rate', value=f"{normal_rate:.1f}%")
    with col5:
        st.metric(label='📈 Average BMI', value=f"{avg_bmi:.1f}")
    
    # Additional metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label='👶 Average Age', value=f"{avg_age:.1f} years")
    with col2:
        male_count = df[df['Gender'] == 'Male'].shape[0]
        st.metric(label='👨 Male Participants', value=f"{male_count:,}")
    with col3:
        female_count = df[df['Gender'] == 'Female'].shape[0]
        st.metric(label='👩 Female Participants', value=f"{female_count:,}")
    with col4:
        states_covered = df['State'].nunique()
        st.metric(label='🗺️ States Covered', value=f"{states_covered}")
    with col5:
        family_history_rate = (df[df['family_history_with_overweight'] == 'yes'].shape[0] / total_participants) * 100
        st.metric(label='👨‍👩‍👧‍👦 Family History Rate', value=f"{family_history_rate:.1f}%")
    
    # Section 2: Demographics Overview
    st.header('👥 Demographics Overview')
    st.write('Understanding the demographic composition of our study population')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution histogram
        fig_age = px.histogram(df, x='Age', nbins=30, 
                              title='Age Distribution of Participants',
                              labels={'Age': 'Age (years)', 'count': 'Number of Participants'})
        fig_age.update_layout(showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_counts = df['Gender'].value_counts()
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                           title='Gender Distribution',
                           color_discrete_sequence=['#FF9999', '#66B2FF'])
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Section 3: Obesity Category Distribution
    st.header('🎯 Obesity Category Distribution')
    st.write('Detailed breakdown of obesity categories across the population')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Obesity categories pie chart
        obesity_counts = df['NObeyesdad'].value_counts()
        fig_obesity = px.pie(values=obesity_counts.values, names=obesity_counts.index,
                            title='Distribution of Obesity Categories',
                            color_discrete_sequence=px.colors.qualitative.Set3)
        fig_obesity.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_obesity, use_container_width=True)
    
    with col2:
        # BMI categories
        bmi_counts = df['BMI_Category'].value_counts()
        fig_bmi = px.bar(x=bmi_counts.index, y=bmi_counts.values,
                        title='BMI Category Distribution',
                        labels={'x': 'BMI Category', 'y': 'Count'},
                        color=bmi_counts.values,
                        color_continuous_scale='RdYlBu_r')
        st.plotly_chart(fig_bmi, use_container_width=True)
    
    # Section 4: Geographic Analysis
    st.header('🗺️ Geographic Analysis')
    st.write('Obesity patterns across Malaysian states')
    
    # Calculate obesity rate by state
    state_obesity = df.groupby('State').apply(
        lambda x: (x['NObeyesdad'].str.contains('Obesity', na=False).sum() / len(x)) * 100
    ).reset_index(name='Obesity_Rate')
    
    # Create choropleth map
    fig_map = px.choropleth_mapbox(
        state_obesity,
        geojson=geojson,
        locations='State',
        color='Obesity_Rate',
        color_continuous_scale='Reds',
        featureidkey="properties.name",
        mapbox_style="carto-positron",
        zoom=4.4,
        center={"lat": 4, "lon": 109.5},
        opacity=0.75,
        labels={'Obesity_Rate': 'Obesity Rate (%)'},
        title='Obesity Rate by State'
    )
    fig_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)
    
    # State-wise detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Top states by obesity rate
        top_states = state_obesity.nlargest(10, 'Obesity_Rate')
        fig_top = px.bar(top_states, x='Obesity_Rate', y='State', orientation='h',
                        title='Top 10 States by Obesity Rate',
                        labels={'Obesity_Rate': 'Obesity Rate (%)', 'State': 'State'})
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # Participant distribution by state
        state_counts = df['State'].value_counts().head(10)
        fig_states = px.bar(x=state_counts.values, y=state_counts.index, orientation='h',
                           title='Top 10 States by Number of Participants',
                           labels={'x': 'Number of Participants', 'y': 'State'})
        fig_states.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_states, use_container_width=True)
    
    # Section 5: Age and Gender Analysis
    st.header('👨‍👩‍👧‍👦 Age and Gender Analysis')
    st.write('How obesity patterns vary across age groups and genders')
    
    # Age group analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Obesity by age group
        age_obesity = df.groupby('Age_Group')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_age_obesity = px.bar(age_obesity, x='Age_Group', y='Obesity_Rate',
                                title='Obesity Rate by Age Group',
                                labels={'Age_Group': 'Age Group', 'Obesity_Rate': 'Obesity Rate (%)'},
                                color='Obesity_Rate',
                                color_continuous_scale='Reds')
        st.plotly_chart(fig_age_obesity, use_container_width=True)
    
    with col2:
        # Gender comparison across obesity categories
        gender_obesity = pd.crosstab(df['Gender'], df['NObeyesdad'], normalize='index') * 100
        fig_gender_obesity = px.bar(gender_obesity, 
                                   title='Obesity Categories by Gender (%)',
                                   labels={'value': 'Percentage', 'index': 'Gender'},
                                   barmode='group')
        st.plotly_chart(fig_gender_obesity, use_container_width=True)
    
    # Detailed age-gender heatmap
    age_gender_crosstab = pd.crosstab(df['Age_Group'], df['Gender'])
    fig_heatmap = px.imshow(age_gender_crosstab.values,
                           x=age_gender_crosstab.columns,
                           y=age_gender_crosstab.index,
                           title='Participant Distribution: Age Group vs Gender',
                           labels={'x': 'Gender', 'y': 'Age Group', 'color': 'Count'},
                           color_continuous_scale='Blues')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Section 6: Physical Measurements Analysis
    st.header('📏 Physical Measurements Analysis')
    st.write('Analysis of height, weight, and BMI patterns')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Height distribution
        fig_height = px.histogram(df, x='Height', nbins=30,
                                 title='Height Distribution',
                                 labels={'Height': 'Height (m)', 'count': 'Count'})
        st.plotly_chart(fig_height, use_container_width=True)
    
    with col2:
        # Weight distribution
        fig_weight = px.histogram(df, x='Weight', nbins=30,
                                 title='Weight Distribution',
                                 labels={'Weight': 'Weight (kg)', 'count': 'Count'})
        st.plotly_chart(fig_weight, use_container_width=True)
    
    with col3:
        # BMI distribution
        fig_bmi_dist = px.histogram(df, x='BMI', nbins=30,
                                   title='BMI Distribution',
                                   labels={'BMI': 'BMI', 'count': 'Count'})
        st.plotly_chart(fig_bmi_dist, use_container_width=True)
    
    # BMI vs Weight scatter plot
    fig_scatter = px.scatter(df, x='Height', y='Weight', color='NObeyesdad',
                            title='Height vs Weight by Obesity Category',
                            labels={'Height': 'Height (m)', 'Weight': 'Weight (kg)'},
                            size='BMI', size_max=10)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Section 7: Lifestyle Factors Analysis
    st.header('🏃‍♂️ Lifestyle Factors Analysis')
    st.write('Impact of various lifestyle factors on obesity')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Physical Activity Frequency (FAF) analysis
        faf_obesity = df.groupby('FAF')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_faf = px.bar(faf_obesity, x='FAF', y='Obesity_Rate',
                        title='Obesity Rate by Physical Activity Frequency',
                        labels={'FAF': 'Physical Activity Frequency (days/week)', 'Obesity_Rate': 'Obesity Rate (%)'},
                        color='Obesity_Rate',
                        color_continuous_scale='RdYlBu_r')
        st.plotly_chart(fig_faf, use_container_width=True)
    
    with col2:
        # Technology Use (TUE) analysis
        tue_obesity = df.groupby('TUE')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_tue = px.bar(tue_obesity, x='TUE', y='Obesity_Rate',
                        title='Obesity Rate by Technology Use Time',
                        labels={'TUE': 'Technology Use Time (hours/day)', 'Obesity_Rate': 'Obesity Rate (%)'},
                        color='Obesity_Rate',
                        color_continuous_scale='RdYlBu_r')
        st.plotly_chart(fig_tue, use_container_width=True)
    
    # Section 8: Eating Habits Analysis
    st.header('🍽️ Eating Habits Analysis')
    st.write('How eating patterns influence obesity rates')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Frequent consumption of high caloric food (FAVC)
        favc_counts = df['FAVC'].value_counts()
        fig_favc = px.pie(values=favc_counts.values, names=favc_counts.index,
                         title='Frequent High Caloric Food Consumption',
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        st.plotly_chart(fig_favc, use_container_width=True)
    
    with col2:
        # Frequency of consumption of vegetables (FCVC)
        fcvc_obesity = df.groupby('FCVC')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_fcvc = px.bar(fcvc_obesity, x='FCVC', y='Obesity_Rate',
                         title='Obesity Rate by Vegetable Consumption',
                         labels={'FCVC': 'Vegetable Consumption Frequency', 'Obesity_Rate': 'Obesity Rate (%)'},
                         color='Obesity_Rate',
                         color_continuous_scale='Greens')
        st.plotly_chart(fig_fcvc, use_container_width=True)
    
    with col3:
        # Number of main meals (NCP)
        ncp_obesity = df.groupby('NCP')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_ncp = px.bar(ncp_obesity, x='NCP', y='Obesity_Rate',
                        title='Obesity Rate by Number of Main Meals',
                        labels={'NCP': 'Number of Main Meals', 'Obesity_Rate': 'Obesity Rate (%)'},
                        color='Obesity_Rate',
                        color_continuous_scale='Blues')
        st.plotly_chart(fig_ncp, use_container_width=True)
    
    # Water consumption analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily water consumption (CH2O)
        ch2o_obesity = df.groupby('CH2O')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_ch2o = px.bar(ch2o_obesity, x='CH2O', y='Obesity_Rate',
                         title='Obesity Rate by Daily Water Consumption',
                         labels={'CH2O': 'Daily Water Consumption (L)', 'Obesity_Rate': 'Obesity Rate (%)'},
                         color='Obesity_Rate',
                         color_continuous_scale='Blues')
        st.plotly_chart(fig_ch2o, use_container_width=True)
    
    with col2:
        # Consumption of food between meals (CAEC)
        caec_counts = df['CAEC'].value_counts()
        fig_caec = px.bar(x=caec_counts.index, y=caec_counts.values,
                         title='Food Consumption Between Meals',
                         labels={'x': 'Frequency', 'y': 'Count'},
                         color=caec_counts.values,
                         color_continuous_scale='Oranges')
        st.plotly_chart(fig_caec, use_container_width=True)
    
    # Section 9: Physical Activity Analysis
    st.header('💪 Physical Activity Analysis')
    st.write('Relationship between physical activity and obesity')
    
    # Create physical activity levels
    df['Activity_Level'] = pd.cut(df['FAF'], 
                                 bins=[-1, 0, 1, 2, 3], 
                                 labels=['Sedentary', 'Low', 'Moderate', 'High'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Activity level distribution
        activity_counts = df['Activity_Level'].value_counts()
        fig_activity = px.pie(values=activity_counts.values, names=activity_counts.index,
                             title='Physical Activity Level Distribution',
                             color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_activity, use_container_width=True)
    
    with col2:
        # Obesity rate by activity level
        activity_obesity = df.groupby('Activity_Level')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_activity_obesity = px.bar(activity_obesity, x='Activity_Level', y='Obesity_Rate',
                                     title='Obesity Rate by Activity Level',
                                     labels={'Activity_Level': 'Activity Level', 'Obesity_Rate': 'Obesity Rate (%)'},
                                     color='Obesity_Rate',
                                     color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_activity_obesity, use_container_width=True)
    
    # Section 10: Transportation Patterns
    st.header('🚗 Transportation Patterns')
    st.write('How transportation methods correlate with obesity')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transportation method distribution
        transport_counts = df['MTRANS'].value_counts()
        fig_transport = px.bar(x=transport_counts.values, y=transport_counts.index, orientation='h',
                              title='Transportation Method Distribution',
                              labels={'x': 'Count', 'y': 'Transportation Method'},
                              color=transport_counts.values,
                              color_continuous_scale='Viridis')
        fig_transport.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_transport, use_container_width=True)
    
    with col2:
        # Obesity rate by transportation method
        transport_obesity = df.groupby('MTRANS')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_transport_obesity = px.bar(transport_obesity, x='Obesity_Rate', y='MTRANS', orientation='h',
                                      title='Obesity Rate by Transportation Method',
                                      labels={'Obesity_Rate': 'Obesity Rate (%)', 'MTRANS': 'Transportation Method'},
                                      color='Obesity_Rate',
                                      color_continuous_scale='Reds')
        fig_transport_obesity.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_transport_obesity, use_container_width=True)
    
    # Section 11: Health Behaviors Analysis
    st.header('🚭 Health Behaviors Analysis')
    st.write('Impact of smoking and alcohol consumption on obesity')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Smoking habits
        smoke_counts = df['SMOKE'].value_counts()
        fig_smoke = px.pie(values=smoke_counts.values, names=smoke_counts.index,
                          title='Smoking Habits Distribution',
                          color_discrete_sequence=['#90EE90', '#FF6B6B'])
        st.plotly_chart(fig_smoke, use_container_width=True)
    
    with col2:
        # Alcohol consumption (CALC)
        calc_counts = df['CALC'].value_counts()
        fig_calc = px.pie(values=calc_counts.values, names=calc_counts.index,
                         title='Alcohol Consumption Pattern',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_calc, use_container_width=True)
    
    with col3:
        # Caloric consumption monitoring (SCC)
        scc_counts = df['SCC'].value_counts()
        fig_scc = px.pie(values=scc_counts.values, names=scc_counts.index,
                        title='Calorie Consumption Monitoring',
                        color_discrete_sequence=['#FFB6C1', '#87CEEB'])
        st.plotly_chart(fig_scc, use_container_width=True)
    
    # Section 12: Risk Factors Assessment
    st.header('⚠️ Risk Factors Assessment')
    st.write('Key risk factors associated with obesity')
    
    # Family history impact
    col1, col2 = st.columns(2)
    
    with col1:
        # Family history vs obesity
        family_obesity = pd.crosstab(df['family_history_with_overweight'], 
                                   df['NObeyesdad'].str.contains('Obesity', na=False),
                                   normalize='index') * 100
        
        fig_family = px.bar(family_obesity, 
                           title='Obesity Rate by Family History',
                           labels={'value': 'Percentage', 'index': 'Family History'},
                           color_discrete_sequence=['#87CEEB', '#FF6B6B'])
        fig_family.update_layout(legend_title_text='Has Obesity')
        st.plotly_chart(fig_family, use_container_width=True)
    
    with col2:
        # Multiple risk factors analysis
        df['Risk_Score'] = 0
        df.loc[df['family_history_with_overweight'] == 'yes', 'Risk_Score'] += 1
        df.loc[df['FAVC'] == 'yes', 'Risk_Score'] += 1
        df.loc[df['SMOKE'] == 'yes', 'Risk_Score'] += 1
        df.loc[df['SCC'] == 'no', 'Risk_Score'] += 1
        df.loc[df['FAF'] < 1, 'Risk_Score'] += 1
        
        risk_obesity = df.groupby('Risk_Score')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_risk = px.bar(risk_obesity, x='Risk_Score', y='Obesity_Rate',
                         title='Obesity Rate by Risk Factor Count',
                         labels={'Risk_Score': 'Number of Risk Factors', 'Obesity_Rate': 'Obesity Rate (%)'},
                         color='Obesity_Rate',
                         color_continuous_scale='Reds')
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Prediction Tool Section
    st.header('🔮 Try Our Obesity Prediction Tool')
    st.write('Use machine learning models to predict your obesity level based on lifestyle and demographic factors')
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button('🚀 Predict My Obesity Level', type="primary", use_container_width=True):
            st.switch_page("pages/5_Prediction.py")
    
    st.markdown("---")
    
    # Topic selection (you would need to create these image files)
    try:
        from streamlit_image_select import image_select
        
        img = image_select(
            label="Select a topic to dive deeper into specific obesity insights:",
            images=[
                "images/demographics_btn.png",
                "images/lifestyle_btn.png", 
                "images/health_btn.png",
                "images/geographic_btn.png"
            ],
            captions=["Demographics Analysis", "Lifestyle Factors", "Health Behaviors", "Geographic Patterns"],
        )
        
        # Navigation buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        if col3.button('🚀 Explore Topic', type="primary"):
            if img == "images/demographics_btn.png":
                st.switch_page("pages/1_Demographics.py")
            elif img == "images/lifestyle_btn.png":
                st.switch_page("pages/2_Lifestyle.py")
            elif img == "images/health_btn.png":
                st.switch_page("pages/3_Health_Analysis.py")
            elif img == "images/geographic_btn.png":
                st.switch_page("pages/4_Geographic.py")
                
    except ImportError:
        st.info("Install streamlit-image-select for topic selection: `pip install streamlit-image-select`")
        
        # Alternative topic selection
        topic = st.selectbox(
            "Choose a topic to explore:",
            ["Demographics Analysis", "Lifestyle Factors", "Health Behaviors", "Geographic Patterns", "Obesity Prediction"]
        )
        
        if st.button('🚀 Explore Selected Topic', type="primary"):
            if topic == "Demographics Analysis":
                st.switch_page("pages/1_👥_Demographics.py")
            elif topic == "Lifestyle Factors":
                st.switch_page("pages/2_🏃_Lifestyle.py")
            elif topic == "Health Behaviors":
                st.switch_page("pages/3_🏥_Health_Analysis.py")
            elif topic == "Geographic Patterns":
                st.switch_page("pages/4_🗺️_Geographic.py")
            elif topic == "Obesity Prediction":
                st.switch_page("pages/5_Prediction.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Malaysian Obesity Analytics Dashboard | Data-driven insights for healthier communities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
