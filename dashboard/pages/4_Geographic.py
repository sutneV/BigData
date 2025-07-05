import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

@st.cache_data
def load_data():
    df = pd.read_csv('datasets/Unprocessed_Obesity_Dataset.csv')
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    return df

@st.cache_data
def load_geojson():
    with open('./map_json/malaysia.geojson') as f:
        return json.load(f)

def main():
    st.set_page_config(page_title="Geographic Patterns", layout="wide")
    
    st.title('🗺️ Geographic Patterns Analysis')
    st.markdown("Spatial analysis of obesity patterns across Malaysian states")
    
    df = load_data()
    geojson = load_geojson()
    
    # State-level analysis
    state_analysis = df.groupby('State').agg({
        'NObeyesdad': lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100,
        'BMI': 'mean',
        'Age': 'mean',
        'FAF': 'mean',
        'TUE': 'mean',
        'Weight': 'mean',
        'Height': 'mean'
    }).round(2)
    
    state_analysis.columns = ['Obesity_Rate', 'Avg_BMI', 'Avg_Age', 'Avg_Activity', 
                             'Avg_ScreenTime', 'Avg_Weight', 'Avg_Height']
    state_analysis = state_analysis.reset_index()
    
    # Sidebar for state comparison
    with st.sidebar:
        st.header('State Comparison')
        selected_states = st.multiselect('Compare States', 
                                       df['State'].unique(), 
                                       default=df['State'].unique()[:5])
        
        metric_to_compare = st.selectbox('Select Metric', 
                                       ['Obesity_Rate', 'Avg_BMI', 'Avg_Age', 
                                        'Avg_Activity', 'Avg_ScreenTime'])
    
    # Section 1: Interactive Maps
    st.header('🌏 Interactive Obesity Maps')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Obesity rate choropleth
        fig_obesity_map = px.choropleth_mapbox(
            state_analysis,
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
        fig_obesity_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        st.plotly_chart(fig_obesity_map, use_container_width=True)
    
    with col2:
        # BMI choropleth
        fig_bmi_map = px.choropleth_mapbox(
            state_analysis,
            geojson=geojson,
            locations='State',
            color='Avg_BMI',
            color_continuous_scale='Blues',
            featureidkey="properties.name",
            mapbox_style="carto-positron",
            zoom=4.4,
            center={"lat": 4, "lon": 109.5},
            opacity=0.75,
            labels={'Avg_BMI': 'Average BMI'},
            title='Average BMI by State'
        )
        fig_bmi_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        st.plotly_chart(fig_bmi_map, use_container_width=True)
    
    # Section 2: State Rankings
    st.header('🏆 State Rankings')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Top states by obesity rate
        top_obesity = state_analysis.nlargest(10, 'Obesity_Rate')
        fig_top_obesity = px.bar(top_obesity, x='Obesity_Rate', y='State', orientation='h',
                                title='Top 10 States by Obesity Rate',
                                color='Obesity_Rate', color_continuous_scale='Reds')
        fig_top_obesity.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top_obesity, use_container_width=True)
    
    with col2:
        # Lowest obesity rates
        low_obesity = state_analysis.nsmallest(10, 'Obesity_Rate')
        fig_low_obesity = px.bar(low_obesity, x='Obesity_Rate', y='State', orientation='h',
                                title='Top 10 States with Lowest Obesity Rate',
                                color='Obesity_Rate', color_continuous_scale='Greens')
        fig_low_obesity.update_layout(yaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_low_obesity, use_container_width=True)
    
    with col3:
        # States by average BMI
        high_bmi = state_analysis.nlargest(10, 'Avg_BMI')
        fig_high_bmi = px.bar(high_bmi, x='Avg_BMI', y='State', orientation='h',
                             title='Top 10 States by Average BMI',
                             color='Avg_BMI', color_continuous_scale='Blues')
        fig_high_bmi.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_high_bmi, use_container_width=True)
    
    # Section 3: Regional Patterns
    st.header('🏞️ Regional Pattern Analysis')
    
    # Define regions (you can customize this based on Malaysian geography)
    regions = {
        'Central': ['Selangor', 'Wilayah Persekutuan Kuala Lumpur', 'Wilayah Persekutuan Putrajaya'],
        'Northern': ['Kedah', 'Perlis', 'Pulau Pinang', 'Perak'],
        'Southern': ['Johor', 'Melaka', 'Negeri Sembilan'],
        'Eastern': ['Kelantan', 'Terengganu', 'Pahang'],
        'East Malaysia': ['Sabah', 'Sarawak', 'Wilayah Persekutuan Labuan']
    }
    
    # Add region column
    def get_region(state):
        for region, states in regions.items():
            if state in states:
                return region
        return 'Other'
    
    df['Region'] = df['State'].apply(get_region)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional obesity rates
        regional_obesity = df.groupby('Region')['NObeyesdad'].apply(
            lambda x: (x.str.contains('Obesity', na=False).sum() / len(x)) * 100
        ).reset_index(name='Obesity_Rate')
        
        fig_regional = px.bar(regional_obesity, x='Region', y='Obesity_Rate',
                             title='Obesity Rate by Region',
                             color='Obesity_Rate', color_continuous_scale='Reds')
        st.plotly_chart(fig_regional, use_container_width=True)
    
    with col2:
        # Regional BMI comparison
        fig_regional_bmi = px.box(df, x='Region', y='BMI', color='Region',
                                 title='BMI Distribution by Region')
        st.plotly_chart(fig_regional_bmi, use_container_width=True)
    
    # Section 4: State Comparison Tool
    st.header('⚖️ State Comparison Tool')
    
    if selected_states:
        comparison_data = state_analysis[state_analysis['State'].isin(selected_states)]
        
        # Radar chart for multi-dimensional comparison
        fig_radar = go.Figure()
        
        metrics = ['Obesity_Rate', 'Avg_BMI', 'Avg_Activity', 'Avg_ScreenTime']
        
        for state in selected_states:
            state_data = comparison_data[comparison_data['State'] == state]
            if not state_data.empty:
                values = [state_data[metric].iloc[0] for metric in metrics]
                values.append(values[0])  # Close the radar chart
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=state
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            title="State Comparison - Multi-dimensional Analysis",
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed comparison table
        st.subheader('Detailed State Comparison')
        st.dataframe(comparison_data.set_index('State'))
    
    # Section 5: Urban vs Rural Analysis (if data available)
    st.header('🏙️ Geographic Insights')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation between lifestyle factors and geography
        geo_lifestyle = df.groupby('State').agg({
            'FAF': 'mean',
            'TUE': 'mean',
            'MTRANS': lambda x: (x == 'Public_Transportation').sum() / len(x) * 100
        }).round(2)
        geo_lifestyle.columns = ['Avg_Activity', 'Avg_ScreenTime', 'Public_Transport_Usage']
        
        fig_lifestyle_geo = px.scatter(geo_lifestyle, x='Avg_Activity', y='Public_Transport_Usage',
                                      size='Avg_ScreenTime',
                                      title='Lifestyle Patterns by State',
                                      labels={'Avg_Activity': 'Average Physical Activity',
                                             'Public_Transport_Usage': 'Public Transport Usage (%)'})
        st.plotly_chart(fig_lifestyle_geo, use_container_width=True)
    
    with col2:
        # State diversity analysis
        state_diversity = df.groupby('State').agg({
            'Gender': lambda x: x.nunique(),
            'Age': lambda x: x.max() - x.min(),
            'NObeyesdad': lambda x: x.nunique()
        })
        state_diversity.columns = ['Gender_Diversity', 'Age_Range', 'Obesity_Categories']
        
        fig_diversity = px.scatter(state_diversity, x='Age_Range', y='Obesity_Categories',
                                  size='Gender_Diversity',
                                  title='State Population Diversity',
                                  labels={'Age_Range': 'Age Range', 
                                         'Obesity_Categories': 'Number of Obesity Categories'})
        st.plotly_chart(fig_diversity, use_container_width=True)
    
    # Geographic insights summary
    st.header('📊 Key Geographic Insights')
    
    insights = []
    
    # Find highest and lowest obesity rate states
    highest_obesity_state = state_analysis.loc[state_analysis['Obesity_Rate'].idxmax()]
    lowest_obesity_state = state_analysis.loc[state_analysis['Obesity_Rate'].idxmin()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Highest Obesity Rate', 
                 f"{highest_obesity_state['State']}", 
                 f"{highest_obesity_state['Obesity_Rate']:.1f}%")
    
    with col2:
        st.metric('Lowest Obesity Rate', 
                 f"{lowest_obesity_state['State']}", 
                 f"{lowest_obesity_state['Obesity_Rate']:.1f}%")
    
    with col3:
        obesity_variation = state_analysis['Obesity_Rate'].std()
        st.metric('State Variation (Std Dev)', f"{obesity_variation:.1f}%")

if __name__ == "__main__":
    main()