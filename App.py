import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set page configuration
st.set_page_config(
    page_title="Patient Clustering and Risk Analysis",
    page_icon="🏥",
    layout="wide"
)

# Load the trained models
def load_models():
    try:
        with open("kmeans_model.pkl", "rb") as f:
            kmeans_model = pickle.load(f)
        
        with open("random_forest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
            
        with open("xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
            
        return kmeans_model, rf_model, xgb_model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the models are properly trained and saved.")
        return None, None, None

# Function to handle outliers using IQR method
def handle_outliers(df, column, method='cap'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if method == 'cap':
        # Cap the outliers at the boundaries
        df.loc[df[column] < lower_bound, column] = lower_bound
        df.loc[df[column] > upper_bound, column] = upper_bound
    elif method == 'remove':
        # Set outliers as NaN
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
    
    return df

# Function to fix physiologically impossible values
def fix_impossible_values(df):
    # Blood pressure can't be 0
    df.loc[df['blood_pressure'] == 0, 'blood_pressure'] = np.nan

    # Fix other impossible values based on medical knowledge
    df.loc[df['cholesterol'] > 600, 'cholesterol'] = np.nan  # Extremely rare to be above 600
    df.loc[df['bmi'] < 10, 'bmi'] = np.nan  # BMI below 10 is life-threatening

    return df

# Function to create feature engineering
def create_features(df):
    # Age groups - clinical risk often correlates with age brackets
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 30, 45, 65, 85, 100],
                             labels=['young_adult', 'adult', 'middle_aged', 'senior', 'elderly'])

    # BMI categories according to WHO standards
    df['bmi_category'] = pd.cut(df['bmi'],
                                bins=[0, 18.5, 25, 30, 35, 100],
                                labels=['underweight', 'normal', 'overweight', 'obese', 'severely_obese'])

    # Blood pressure categories based on clinical guidelines
    df['bp_category'] = pd.cut(df['blood_pressure'],
                              bins=[0, 120, 140, 160, 180, 300],
                              labels=['normal', 'elevated', 'hypertension1', 'hypertension2', 'hypertensive_crisis'])

    # Cholesterol levels category
    df['cholesterol_category'] = pd.cut(df['cholesterol'],
                                       bins=[0, 200, 240, 300],
                                       labels=['normal', 'borderline', 'high'])

    # Create combination risk factors
    # Cardiovascular risk - combine multiple factors
    df['cv_risk_count'] = ((df['heart_disease'] == 1).astype(int) +
                          (df['hypertension'] == 1).astype(int) +
                          (df['cholesterol'] > 240).astype(int) +
                          (df['smoking_status'] == 'Smoker').astype(int))

    # Metabolic risk - diabetes indicators
    df['diabetes_risk'] = np.where(
        (df['plasma_glucose'] > 125) & (df['bmi'] > 30), 'high',
        np.where((df['plasma_glucose'] > 100) | (df['bmi'] > 25), 'moderate', 'low')
    )

    # Age-related risk factor
    df['age_risk'] = np.where(df['age'] > 65, 'high',
                             np.where(df['age'] > 45, 'moderate', 'low'))

    # Heart rate abnormality
    df['hr_abnormal'] = np.where(
        (df['max_heart_rate'] < 60) | (df['max_heart_rate'] > 100), 1, 0
    )

    # Chest pain severity (assuming type 1 is most severe, type 4 is least severe)
    df['severe_chest_pain'] = np.where(df['chest_pain_type'] <= 2, 1, 0)

    # Comprehensive risk score
    df['risk_score'] = (
        # Cardiovascular components
        df['cv_risk_count'] * 2 +
        (df['severe_chest_pain'] * 3) +
        (df['hr_abnormal'] * 2) +

        # Convert BP category to numeric and weight it
        pd.factorize(df['bp_category'])[0] * 0.5 +

        # Convert cholesterol category to numeric and weight it
        pd.factorize(df['cholesterol_category'])[0] * 0.5 +

        # Metabolic components
        (pd.factorize(df['diabetes_risk'])[0] * 0.5) +

        # Age component
        (pd.factorize(df['age_risk'])[0] * 0.5)
    )
    
    return df

# Function to preprocess data
def preprocess_data(df):
    # Fix impossible values
    df = fix_impossible_values(df)
    
    # Define numerical columns for outlier handling
    numerical_columns = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate',
                        'plasma_glucose', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree']
    
    # Handle outliers
    for column in numerical_columns:
        if column in df.columns:
            df = handle_outliers(df, column, method='cap')
    
    # Handle missing values
    for column in numerical_columns:
        if column in df.columns:
            df[column].fillna(df[column].median(), inplace=True)
    
    # Handle missing values in categorical features
    categorical_features = ['gender', 'chest_pain_type', 'residence_type', 'smoking_status']
    for feature in categorical_features:
        if feature in df.columns:
            df[feature].fillna(df[feature].mode()[0], inplace=True)
    
    # Create engineered features
    df = create_features(df)
    
    return df

# Display the exact features used to train the model
def display_model_features(kmeans_model):
    st.subheader("Model Information")
    
    # Try to extract n_features_in_ attribute (sklearn 0.24+)
    if hasattr(kmeans_model, 'n_features_in_'):
        st.write(f"The KMeans model was trained with {kmeans_model.n_features_in_} features.")
    else:
        st.write("Could not determine the exact number of features used for training.")
    
    # Display cluster centers shape
    if hasattr(kmeans_model, 'cluster_centers_'):
        n_clusters, n_features = kmeans_model.cluster_centers_.shape
        st.write(f"Model has {n_clusters} clusters with centers of shape {kmeans_model.cluster_centers_.shape}")

# Function to ensure feature compatibility with model
def align_features_with_model(df, kmeans_model):
    """Ensure the features match what the model expects"""
    
    # Get model expected features count
    if hasattr(kmeans_model, 'n_features_in_'):
        expected_feature_count = kmeans_model.n_features_in_
    else:
        expected_feature_count = kmeans_model.cluster_centers_.shape[1]
    
    # Get current features
    numerical_features = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate',
                         'bmi', 'diabetes_pedigree', 'plasma_glucose',
                         'skin_thickness', 'insulin', 'cv_risk_count', 'risk_score']
    
    binary_features = ['exercise_angina', 'hypertension', 'heart_disease',
                      'severe_chest_pain', 'hr_abnormal']
    
    # Get available features in data
    available_features = [f for f in numerical_features + binary_features if f in df.columns]
    
    # Display diagnostic information
    st.write(f"Available features: {len(available_features)} | Model expects: {expected_feature_count}")
    
    # If we have too few features, add dummy ones
    if len(available_features) < expected_feature_count:
        st.warning(f"Missing {expected_feature_count - len(available_features)} features expected by the model. Adding placeholder features.")
        # Add dummy features with median values
        for i in range(len(available_features), expected_feature_count):
            feature_name = f"placeholder_feature_{i}"
            df[feature_name] = 0  # Default value
            available_features.append(feature_name)
    
    # If we have too many features, keep only what we need
    if len(available_features) > expected_feature_count:
        st.warning(f"Too many features ({len(available_features)}). Keeping only the first {expected_feature_count}.")
        available_features = available_features[:expected_feature_count]
    
    return df, available_features

# Function to predict cluster for a single patient
def predict_cluster(data, kmeans_model, scaler, features):
    # Extract features for clustering
    X = data[features]
    X_scaled = scaler.transform(X)
    
    # Predict cluster
    cluster = kmeans_model.predict(X_scaled)[0]
    return cluster

# Function to display cluster information
def display_cluster_info(cluster, df):
    st.subheader(f"Cluster {cluster} Characteristics")
    
    # Filter data for this cluster
    cluster_data = df[df['cluster'] == cluster]
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate statistics for the cluster
    cluster_stats = cluster_data[numerical_cols].mean()
    overall_stats = df[numerical_cols].mean()
    
    # Create a comparison dataframe
    comparison = pd.DataFrame({
        'Cluster Mean': cluster_stats,
        'Overall Mean': overall_stats,
        'Difference': cluster_stats - overall_stats
    })
    
    st.dataframe(comparison.style.highlight_max(axis=0, subset=['Difference']))
    
    # Visualize key metrics
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Risk score distribution
    if 'risk_score' in df.columns:
        sns.histplot(data=df, x='risk_score', hue='cluster', kde=True, ax=ax[0])
        ax[0].axvline(x=cluster_data['risk_score'].mean(), color='red', linestyle='--')
        ax[0].set_title('Risk Score Distribution')
    
    # Age vs BMI scatter plot
    if 'age' in df.columns and 'bmi' in df.columns:
        sns.scatterplot(data=df, x='age', y='bmi', hue='cluster', alpha=0.5, ax=ax[1])
        sns.scatterplot(data=cluster_data, x='age', y='bmi', color='red', s=100, marker='X', ax=ax[1])
        ax[1].set_title('Age vs. BMI by Cluster')
    
    st.pyplot(fig)

# Main application
def main():
    st.title("Patient Clustering and Risk Analysis")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Upload", "Patient Analysis", "Cluster Visualization", "About"])
    
    # Load models
    kmeans_model, rf_model, xgb_model = load_models()
    
    if page == "Data Upload":
        st.header("Upload Patient Data")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])
        
        if uploaded_file is not None:
            # Load and display the data
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            
            st.success(f"Data loaded successfully! Shape: {df.shape}")
            
            # Display first few rows
            st.subheader("Preview of the data")
            st.dataframe(df.head())
            
            # Display model information if available
            if kmeans_model is not None:
                display_model_features(kmeans_model)
            
            # Process the data
            if st.button("Preprocess Data"):
                with st.spinner("Processing data..."):
                    processed_df = preprocess_data(df)
                    st.session_state['processed_data'] = processed_df
                    
                    # Display processed data
                    st.subheader("Processed Data Preview")
                    st.dataframe(processed_df.head())
                    
                    # Align features with the model if available
                    if kmeans_model is not None:
                        aligned_df, model_features = align_features_with_model(processed_df, kmeans_model)
                        st.session_state['processed_data'] = aligned_df
                        st.session_state['model_features'] = model_features
                        
                        st.success(f"Data preprocessing completed and features aligned with model!")
                        st.write(f"Using these features for prediction: {model_features}")
                    else:
                        st.success("Data preprocessing completed!")
    
    elif page == "Patient Analysis":
        st.header("Patient Analysis")
        
        if 'processed_data' not in st.session_state:
            st.warning("Please upload and process data first!")
            return
        
        df = st.session_state['processed_data']
        
        # Get model features if they're in the session state
        if 'model_features' in st.session_state:
            model_features = st.session_state['model_features']
        else:
            # Create default features for the model
            numerical_features = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate',
                                'bmi', 'diabetes_pedigree', 'plasma_glucose',
                                'skin_thickness', 'insulin', 'cv_risk_count', 'risk_score']
            binary_features = ['exercise_angina', 'hypertension', 'heart_disease',
                              'severe_chest_pain', 'hr_abnormal']
            
            available_features = [f for f in numerical_features + binary_features if f in df.columns]
            
            # Align with model if available
            if kmeans_model is not None:
                df, model_features = align_features_with_model(df, kmeans_model)
                st.session_state['model_features'] = model_features
            else:
                model_features = available_features
        
        # Create scaler
        scaler = StandardScaler()
        X = df[model_features]
        scaler.fit(X)
        
        # Patient selector
        patient_id = st.selectbox("Select Patient ID", df.index)
        
        # Display patient info
        patient_data = df.loc[patient_id:patient_id]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Basic Information")
            if 'age' in patient_data:
                st.write(f"**Age:** {patient_data['age'].values[0]}")
            if 'gender' in patient_data:
                st.write(f"**Gender:** {patient_data['gender'].values[0]}")
            if 'residence_type' in patient_data:
                st.write(f"**Residence Type:** {patient_data['residence_type'].values[0]}")
                
            st.subheader("Risk Factors")
            if 'hypertension' in patient_data:
                st.write(f"**Hypertension:** {'Yes' if patient_data['hypertension'].values[0] == 1 else 'No'}")
            if 'heart_disease' in patient_data:
                st.write(f"**Heart Disease:** {'Yes' if patient_data['heart_disease'].values[0] == 1 else 'No'}")
            if 'smoking_status' in patient_data:
                st.write(f"**Smoking Status:** {patient_data['smoking_status'].values[0]}")
            if 'diabetes_risk' in patient_data:
                st.write(f"**Diabetes Risk:** {patient_data['diabetes_risk'].values[0]}")
        
        with col2:
            st.subheader("Clinical Measurements")
            if 'blood_pressure' in patient_data:
                st.write(f"**Blood Pressure:** {patient_data['blood_pressure'].values[0]}")
            if 'cholesterol' in patient_data:
                st.write(f"**Cholesterol:** {patient_data['cholesterol'].values[0]}")
            if 'bmi' in patient_data:
                st.write(f"**BMI:** {patient_data['bmi'].values[0]}")
            if 'plasma_glucose' in patient_data:
                st.write(f"**Plasma Glucose:** {patient_data['plasma_glucose'].values[0]}")
            if 'max_heart_rate' in patient_data:
                st.write(f"**Max Heart Rate:** {patient_data['max_heart_rate'].values[0]}")
            
            st.subheader("Risk Assessment")
            if 'risk_score' in patient_data:
                risk_score = patient_data['risk_score'].values[0]
                st.write(f"**Risk Score:** {risk_score:.2f}")
                
                # Risk level interpretation
                if risk_score < 3:
                    risk_level = "Low"
                    color = "green"
                elif risk_score < 6:
                    risk_level = "Moderate"
                    color = "orange"
                else:
                    risk_level = "High"
                    color = "red"
                
                st.markdown(f"**Risk Level:** <span style='color:{color};font-weight:bold'>{risk_level}</span>", unsafe_allow_html=True)
        
        # Predict cluster
        if kmeans_model is not None:
            cluster = predict_cluster(patient_data, kmeans_model, scaler, model_features)
            st.subheader(f"Patient belongs to Cluster {cluster}")
            
            # Show cluster info
            if st.button("Show Cluster Details"):
                # Add cluster labels to the dataset for visualization
                df['cluster'] = kmeans_model.predict(scaler.transform(df[model_features]))
                display_cluster_info(cluster, df)
    
    elif page == "Cluster Visualization":
        st.header("Cluster Visualization")
        
        if 'processed_data' not in st.session_state:
            st.warning("Please upload and process data first!")
            return
        
        df = st.session_state['processed_data']
        
        # Get model features if they're in the session state
        if 'model_features' in st.session_state:
            model_features = st.session_state['model_features']
        else:
            # Create default features for the model
            numerical_features = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate', 
                                'bmi', 'diabetes_pedigree', 'plasma_glucose',
                                'skin_thickness', 'insulin', 'cv_risk_count', 'risk_score']
            binary_features = ['exercise_angina', 'hypertension', 'heart_disease',
                              'severe_chest_pain', 'hr_abnormal']
            
            available_features = [f for f in numerical_features + binary_features if f in df.columns]
            
            # Align with model if available
            if kmeans_model is not None:
                df, model_features = align_features_with_model(df, kmeans_model)
                st.session_state['model_features'] = model_features
            else:
                model_features = available_features
        
        # Create scaler and scale features
        scaler = StandardScaler()
        X = df[model_features]
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering
        if kmeans_model is not None:
            df['cluster'] = kmeans_model.predict(X_scaled)
            
            # PCA for visualization
            n_components = min(2, len(model_features))
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create PCA dataframe
            pca_columns = [f'PCA{i+1}' for i in range(n_components)]
            pca_df = pd.DataFrame(X_pca, columns=pca_columns)
            pca_df['Cluster'] = df['cluster']
            
            # Display PCA scatter plot if we have 2 components
            if n_components == 2:
                st.subheader("PCA Visualization of Patient Clusters")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=50, alpha=0.7, ax=ax)
                plt.title('Patient Clustering Results (PCA Visualization)')
                plt.legend(title='Cluster')
                st.pyplot(fig)
            else:
                st.info("Cannot create PCA visualization with fewer than 2 features.")
            
            # Display cluster sizes
            st.subheader("Cluster Sizes")
            cluster_sizes = df['cluster'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            cluster_sizes.plot(kind='bar', ax=ax)
            plt.title('Number of Patients per Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            # Display cluster profiles
            st.subheader("Cluster Profiles")
            
            # Select only numeric columns for the aggregation
            numeric_df = df.select_dtypes(include=np.number)
            
            # Group by 'cluster' and calculate the mean for numeric columns
            cluster_profiles = numeric_df.groupby('cluster').mean().round(2)
            st.dataframe(cluster_profiles)
            
            # Heatmap of cluster characteristics
            st.subheader("Heatmap of Cluster Characteristics")
            
            feature_subset = ['age', 'blood_pressure', 'cholesterol', 'bmi', 
                             'plasma_glucose', 'risk_score']
            available_features = [f for f in feature_subset if f in cluster_profiles.columns]
            
            if available_features:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(cluster_profiles[available_features], annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
                plt.title('Cluster Characteristics (Mean Values)')
                st.pyplot(fig)
            else:
                st.warning("No suitable features available for heatmap visualization.")
    
    elif page == "About":
        st.header("About This Application")
        
        st.markdown("""
        #### Patient Clustering and Risk Analysis Tool
        
        This application uses machine learning to analyze patient health data and identify patterns through clustering. 
        The system helps healthcare providers understand patient risk profiles and potential health outcomes.
        
        #### Features:
        
        - **Data Preprocessing**: Handles missing values, outliers, and creates meaningful health-related features
        - **Patient Clustering**: Groups patients with similar health characteristics using K-means clustering
        - **Risk Assessment**: Calculates comprehensive risk scores based on multiple health factors
        - **Visualization**: Provides visual analysis of patient clusters and risk distributions
        
        #### How to Use:
        
        1. Upload a CSV file containing patient health data
        2. Process the data to create necessary features
        3. Analyze individual patients or explore cluster patterns
        4. Use insights to inform healthcare decisions
        
        #### Required Data Format:
        
        The application expects a CSV file with the following main columns:
        - `age`, `gender`, `blood_pressure`, `cholesterol`, `max_heart_rate`
        - `bmi`, `plasma_glucose`, `skin_thickness`, `insulin`
        - `hypertension`, `heart_disease`, `smoking_status`, etc.
        
        For optimal results, ensure your data includes these key health metrics.
        """)

if __name__ == "__main__":
    main()