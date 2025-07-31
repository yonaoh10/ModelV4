import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import os
from xgboost_loan_default_model import LoanDefaultPredictor

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def load_model():
    """Load the trained model"""
    try:
        # Find the most recent model file
        model_files = [f for f in os.listdir('.') if f.startswith('xgboost_loan_model_') and f.endswith('.pkl')]
        if not model_files:
            st.error("No trained model found. Please train the model first by running xgboost_loan_default_model.py")
            return False
        
        # Load the most recent model
        latest_model = sorted(model_files)[-1]
        predictor = LoanDefaultPredictor()
        predictor.load_model(latest_model)
        st.session_state.predictor = predictor
        st.session_state.model_loaded = True
        st.success(f"Model loaded successfully: {latest_model}")
        return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

def create_gauge_chart(probability, title="Default Probability"):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 60], 'color': 'yellow'},
                {'range': [60, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        font={'color': "darkblue"},
        height=300
    )
    
    return fig

def get_risk_style(risk_level):
    """Get CSS class for risk level styling"""
    if risk_level == "Low Risk":
        return "risk-low"
    elif risk_level == "Medium Risk":
        return "risk-medium"
    else:
        return "risk-high"

def main():
    # Main title
    st.markdown('<h1 class="main-header">💰 Loan Default Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar for model management
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🔧 Model Management</h2>', unsafe_allow_html=True)
        
        if st.button("Load Trained Model", type="primary"):
            load_model()
        
        if st.session_state.model_loaded:
            st.success("✅ Model is loaded and ready!")
        else:
            st.warning("⚠️ No model loaded. Please load a trained model first.")
            st.info("💡 Tip: Run the training script first to create a model.")
    
    # Main application
    if not st.session_state.model_loaded:
        st.info("Please load a trained model using the sidebar to start making predictions.")
        
        # Show model training instructions
        st.markdown('<h2 class="sub-header">📚 Getting Started</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        **To use this application:**
        
        1. **Train the Model**: Run the training script to create a model
           ```python
           python xgboost_loan_default_model.py
           ```
        
        2. **Load the Model**: Use the "Load Trained Model" button in the sidebar
        
        3. **Make Predictions**: Enter loan application details to get predictions
        
        **Features of this system:**
        - 🎯 Accurate XGBoost-based predictions
        - 📊 Comprehensive risk assessment
        - 🔍 Feature importance analysis
        - 📈 Interactive visualizations
        - 💾 Model persistence and loading
        """)
        
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🏠 Single Prediction", "📊 Batch Analysis", "📈 Model Insights"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🏠 Single Loan Prediction</h2>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("loan_application"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("👤 Personal Information")
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                gender = st.selectbox("Gender", ["Male", "Female"])
                marital_status = st.selectbox("Marital Status", 
                                            ["Single", "Married", "Divorced", "Widowed"])
                num_dependents = st.number_input("Number of Dependents", 
                                               min_value=0, max_value=10, value=0)
                employment_length = st.number_input("Employment Length (years)", 
                                                  min_value=0, max_value=50, value=5)
            
            with col2:
                st.subheader("💰 Financial Information")
                annual_income = st.number_input("Annual Income ($)", 
                                              min_value=10000, max_value=10000000, value=50000)
                credit_score = st.number_input("Credit Score", 
                                             min_value=300, max_value=850, value=700)
                debt_to_income = st.number_input("Debt-to-Income Ratio", 
                                               min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                num_open_credit_lines = st.number_input("Number of Open Credit Lines", 
                                                       min_value=0, max_value=50, value=5)
                num_derogatory_marks = st.number_input("Number of Derogatory Marks", 
                                                     min_value=0, max_value=20, value=0)
                months_since_last_delinquency = st.number_input(
                    "Months Since Last Delinquency", 
                    min_value=0, max_value=1000, value=50,
                    help="Enter 0 if no previous delinquency"
                )
            
            with col3:
                st.subheader("🏦 Loan Information")
                loan_amount = st.number_input("Loan Amount ($)", 
                                            min_value=1000, max_value=1000000, value=25000)
                interest_rate = st.number_input("Interest Rate (%)", 
                                              min_value=0.0, max_value=30.0, value=10.0, step=0.1)
                term_months = st.selectbox("Loan Term (months)", [36, 60])
                loan_purpose = st.selectbox("Loan Purpose", [
                    "debt_consolidation", "home_improvement", "credit_card", 
                    "major_purchase", "car", "small_business", "other"
                ])
                home_ownership = st.selectbox("Home Ownership", 
                                            ["RENT", "MORTGAGE", "OWN", "OTHER"])
                state = st.selectbox("State", [
                    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
                ], index=12)  # Default to IL
                issue_date = st.date_input("Issue Date", value=date.today())
            
            submitted = st.form_submit_button("🔮 Predict Default Risk", type="primary")
            
            if submitted:
                # Prepare loan data
                loan_data = {
                    'age': age,
                    'gender': gender,
                    'marital_status': marital_status,
                    'num_dependents': num_dependents,
                    'employment_length': employment_length,
                    'annual_income': annual_income,
                    'credit_score': credit_score,
                    'loan_amount': loan_amount,
                    'interest_rate': interest_rate,
                    'term_months': term_months,
                    'loan_purpose': loan_purpose,
                    'home_ownership': home_ownership,
                    'state': state,
                    'debt_to_income': debt_to_income,
                    'num_open_credit_lines': num_open_credit_lines,
                    'num_derogatory_marks': num_derogatory_marks,
                    'months_since_last_delinquency': months_since_last_delinquency,
                    'issue_date': issue_date.strftime('%Y-%m-%d')
                }
                
                try:
                    # Make prediction
                    prediction = st.session_state.predictor.predict_single_loan(loan_data)
                    
                    # Display results
                    st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gauge chart
                        fig = create_gauge_chart(prediction['default_probability'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk assessment
                        risk_class = get_risk_style(prediction['risk_level'])
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Risk Assessment</h3>
                            <div class="{risk_class}">
                                {prediction['risk_level']}
                            </div>
                            <p><strong>Default Probability:</strong> {prediction['default_probability']:.2%}</p>
                            <p><strong>Recommendation:</strong> {prediction['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional metrics
                        st.metric("Default Probability", f"{prediction['default_probability']:.2%}")
                        st.metric("Risk Level", prediction['risk_level'])
                        
                        # Color-coded recommendation
                        if prediction['recommendation'] == 'Approve':
                            st.success(f"✅ Recommendation: {prediction['recommendation']}")
                        else:
                            st.error(f"❌ Recommendation: {prediction['recommendation']}")
                    
                    # Risk factors explanation
                    st.markdown('<h3 class="sub-header">🔍 Risk Factor Analysis</h3>', unsafe_allow_html=True)
                    
                    risk_factors = []
                    if credit_score < 650:
                        risk_factors.append("🔴 Low credit score")
                    if debt_to_income > 0.4:
                        risk_factors.append("🔴 High debt-to-income ratio")
                    if num_derogatory_marks > 0:
                        risk_factors.append("🔴 Derogatory marks present")
                    if loan_amount / annual_income > 0.5:
                        risk_factors.append("🔴 High loan-to-income ratio")
                    if employment_length < 2:
                        risk_factors.append("🟡 Short employment history")
                    
                    positive_factors = []
                    if credit_score > 750:
                        positive_factors.append("🟢 Excellent credit score")
                    if debt_to_income < 0.2:
                        positive_factors.append("🟢 Low debt-to-income ratio")
                    if employment_length > 10:
                        positive_factors.append("🟢 Stable employment history")
                    if home_ownership in ['OWN', 'MORTGAGE']:
                        positive_factors.append("🟢 Homeowner")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("⚠️ Risk Factors")
                        if risk_factors:
                            for factor in risk_factors:
                                st.write(factor)
                        else:
                            st.write("✅ No major risk factors identified")
                    
                    with col2:
                        st.subheader("✅ Positive Factors")
                        if positive_factors:
                            for factor in positive_factors:
                                st.write(factor)
                        else:
                            st.write("⚠️ No strong positive factors identified")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📊 Batch Analysis</h2>', unsafe_allow_html=True)
        
        st.info("Upload a CSV file with loan applications for batch processing")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Process Batch Predictions"):
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    for i, row in df.iterrows():
                        try:
                            loan_data = row.to_dict()
                            prediction = st.session_state.predictor.predict_single_loan(loan_data)
                            predictions.append({
                                'index': i,
                                'default_probability': prediction['default_probability'],
                                'risk_level': prediction['risk_level'],
                                'recommendation': prediction['recommendation']
                            })
                            progress_bar.progress((i + 1) / len(df))
                        except Exception as e:
                            st.warning(f"Error processing row {i}: {str(e)}")
                    
                    if predictions:
                        results_df = pd.DataFrame(predictions)
                        
                        # Summary statistics
                        st.subheader("📈 Batch Results Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Applications", len(results_df))
                        with col2:
                            approved = len(results_df[results_df['recommendation'] == 'Approve'])
                            st.metric("Approved", approved)
                        with col3:
                            rejected = len(results_df[results_df['recommendation'] == 'Reject'])
                            st.metric("Rejected", rejected)
                        with col4:
                            avg_prob = results_df['default_probability'].mean()
                            st.metric("Avg Default Prob", f"{avg_prob:.2%}")
                        
                        # Risk distribution
                        risk_counts = results_df['risk_level'].value_counts()
                        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                   title="Risk Level Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">📈 Model Insights</h2>', unsafe_allow_html=True)
        
        # Model information
        if st.session_state.model_loaded:
            st.success("✅ Model is loaded and operational")
            
            # Feature importance visualization
            if hasattr(st.session_state.predictor.model, 'get_score'):
                importance = st.session_state.predictor.model.get_score(importance_type='gain')
                
                if importance:
                    # Convert to DataFrame for easier handling
                    importance_df = pd.DataFrame(
                        list(importance.items()), 
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=True)
                    
                    # Top 20 features
                    top_features = importance_df.tail(20)
                    
                    fig = px.bar(
                        top_features, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Top 20 Most Important Features',
                        labels={'Importance': 'Importance Score', 'Feature': 'Features'}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Model statistics
            st.subheader("📊 Model Statistics")
            
            model_info = {
                "Model Type": "XGBoost Classifier",
                "Number of Features": len(st.session_state.predictor.feature_names),
                "Random State": st.session_state.predictor.random_state,
                "Preprocessing": "StandardScaler + Label Encoding"
            }
            
            for key, value in model_info.items():
                st.write(f"**{key}:** {value}")
            
            # Feature list
            with st.expander("📋 View All Features"):
                feature_df = pd.DataFrame({
                    'Feature': st.session_state.predictor.feature_names,
                    'Index': range(len(st.session_state.predictor.feature_names))
                })
                st.dataframe(feature_df, use_container_width=True)
        
        else:
            st.warning("No model loaded. Please load a model first.")

if __name__ == "__main__":
    main()