import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data():
    df = pd.read_csv('datasets/Unprocessed_Obesity_Dataset.csv')
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Activity_Level'] = pd.cut(df['FAF'], bins=[-1, 0, 1, 2, 3], 
                                 labels=['Sedentary', 'Low', 'Moderate', 'High'])
    df['Screen_Time_Level'] = pd.cut(df['TUE'], bins=[-1, 1, 2, 3, 10], 
                                    labels=['Low', 'Moderate', 'High', 'Very High'])
    return df

def main():
    st.set_page_config(page_title="Lifestyle Factors", layout="wide")
    
    st.title('🏃 Lifestyle Factors Analysis')
    st.markdown("Comprehensive analysis of lifestyle patterns and their impact on obesity")
    
    df = load_data()
    
    # Sidebar filters
    with st.sidebar:
        st.header('Lifestyle Filters')
        activity_levels = st.multiselect('Activity Levels', df['Activity_Level'].unique(), 
                                       default=df['Activity_Level'].unique())
        screen_levels = st.multiselect('Screen Time Levels', df['Screen_Time_Level'].unique(),
                                     default=df['Screen_Time_Level'].unique())
    
    filtered_df = df[(df['Activity_Level'].isin(activity_levels)) & 
                     (df['Screen_Time_Level'].isin(screen_levels))]
    
    # Section 1: Physical Activity Analysis
    st.header('💪 Physical Activity Patterns')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Activity frequency distribution
        fig_activity = px.pie(filtered_df, names='Activity_Level',
                             title='Physical Activity Level Distribution')
        st.plotly_chart(fig_activity, use_container_width=True)
    
    with col2:
        # Obesity rate by activity level
        activity_obesity = filtered_df.groupby('Activity_Level')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_activity_obesity = px.bar(activity_obesity, x='Activity_Level', y='Obesity_Rate',
                                     title='Obesity Rate by Activity Level',
                                     color='Obesity_Rate', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_activity_obesity, use_container_width=True)
    
    with col3:
        # BMI vs Physical Activity
        fig_bmi_activity = px.box(filtered_df, x='Activity_Level', y='BMI',
                                 title='BMI Distribution by Activity Level',
                                 color='Activity_Level')
        st.plotly_chart(fig_bmi_activity, use_container_width=True)
    
    # Section 2: Technology Usage Analysis
    st.header('📱 Technology Usage Patterns')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Screen time vs obesity
        screen_obesity = filtered_df.groupby('TUE')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_screen_obesity = px.bar(screen_obesity, x='TUE', y='Obesity_Rate',
                                   title='Obesity Rate by Daily Screen Time (hours)',
                                   color='Obesity_Rate', color_continuous_scale='Reds')
        st.plotly_chart(fig_screen_obesity, use_container_width=True)
    
    with col2:
        # Screen time vs BMI scatter
        fig_screen_bmi = px.scatter(filtered_df, x='TUE', y='BMI', color='Gender',
                                   title='Screen Time vs BMI by Gender',
                                   trendline='ols')
        st.plotly_chart(fig_screen_bmi, use_container_width=True)
    
    # Section 3: Transportation Patterns
    st.header('🚗 Transportation and Mobility')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transportation distribution
        transport_counts = filtered_df['MTRANS'].value_counts()
        fig_transport = px.bar(x=transport_counts.values, y=transport_counts.index, 
                              orientation='h',
                              title='Transportation Method Distribution',
                              color=transport_counts.values, color_continuous_scale='Viridis')
        fig_transport.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_transport, use_container_width=True)
    
    with col2:
        # Transportation vs obesity
        transport_obesity = filtered_df.groupby('MTRANS')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_transport_obesity = px.bar(transport_obesity, x='Obesity_Rate', y='MTRANS',
                                      orientation='h',
                                      title='Obesity Rate by Transportation Method',
                                      color='Obesity_Rate', color_continuous_scale='Reds')
        fig_transport_obesity.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_transport_obesity, use_container_width=True)
    
    # Section 4: Combined Lifestyle Analysis
    st.header('🔄 Combined Lifestyle Factors')
    
    # Create lifestyle score
    filtered_df['Lifestyle_Score'] = 0
    filtered_df.loc[filtered_df['FAF'] >= 3, 'Lifestyle_Score'] += 2  # Good activity
    filtered_df.loc[filtered_df['FAF'] >= 1, 'Lifestyle_Score'] += 1  # Some activity
    filtered_df.loc[filtered_df['TUE'] <= 1, 'Lifestyle_Score'] += 1  # Low screen time
    filtered_df.loc[filtered_df['MTRANS'].isin(['Walking', 'Bike']), 'Lifestyle_Score'] += 1  # Active transport
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lifestyle score distribution
        lifestyle_dist = filtered_df['Lifestyle_Score'].value_counts().sort_index()
        fig_lifestyle = px.bar(x=lifestyle_dist.index, y=lifestyle_dist.values,
                              title='Lifestyle Score Distribution',
                              labels={'x': 'Lifestyle Score (0-5)', 'y': 'Count'},
                              color=lifestyle_dist.values, color_continuous_scale='Blues')
        st.plotly_chart(fig_lifestyle, use_container_width=True)
    
    with col2:
        # Lifestyle score vs obesity
        lifestyle_obesity = filtered_df.groupby('Lifestyle_Score')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_lifestyle_obesity = px.line(lifestyle_obesity, x='Lifestyle_Score', y='Obesity_Rate',
                                       title='Obesity Rate by Lifestyle Score',
                                       markers=True, line_shape='spline')
        st.plotly_chart(fig_lifestyle_obesity, use_container_width=True)
    
    # Lifestyle recommendations
    st.header('💡 Lifestyle Recommendations')
    
    recommendations = {
        'Sedentary': [
            "Start with 15-20 minutes of daily walking",
            "Use stairs instead of elevators",
            "Stand and stretch every hour during work",
            "Join beginner fitness classes"
        ],
        'Low': [
            "Increase to 30 minutes of moderate activity daily",
            "Try recreational sports or dancing",
            "Use active transportation when possible",
            "Set daily step goals (8,000-10,000 steps)"
        ],
        'Moderate': [
            "Maintain current activity levels",
            "Add strength training 2x per week",
            "Try high-intensity interval training",
            "Participate in weekend outdoor activities"
        ],
        'High': [
            "Excellent activity level - maintain it!",
            "Focus on variety to prevent boredom",
            "Consider training for athletic events",
            "Help others adopt active lifestyles"
        ]
    }
    
    for level, recs in recommendations.items():
        with st.expander(f"Recommendations for {level} Activity Level"):
            for i, rec in enumerate(recs, 1):
                st.write(f"{i}. {rec}")

if __name__ == "__main__":
    main()