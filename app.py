import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import time

# Set page 
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("""
This application predicts whether a bank customer will churn (leave the bank) based on their demographic and financial information.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose what you want to do", 
                               ["Predict Single Customer", "Batch Prediction", "Model Training & Evaluation"])

# Load or train model function
@st.cache_resource
def load_or_train_model():
    """Load trained model or train if not available"""
    try:
        # Try to load pre-trained model
        model = keras.models.load_model('churn_model.h5')
        encoder = joblib.load('churn_encoder.joblib')
        scaler = joblib.load('churn_scaler.joblib')
        st.sidebar.success("✅ Pre-trained model loaded successfully!")
        return model, encoder, scaler
    except:
        st.sidebar.info("⏳ No pre-trained model found. Training new model...")
        return train_model()

def train_model():
    """Train the churn prediction model"""
    # training code 
    data = pd.read_csv("Churn_Modelling.csv")
    data.drop(columns=['CustomerId', 'Surname'], inplace=True)

    X = data.drop(columns=['Exited'])
    y = data['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Encoding and scaling
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    categorical_cols = ['Geography', 'Gender']
    encoder.fit(X_train[categorical_cols])

    X_train_encoded = encoder.transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    df_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X_train_final = pd.concat([X_train[numerical_cols].reset_index(drop=True), df_train_encoded.reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test[numerical_cols].reset_index(drop=True), df_test_encoded.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    # Model architecture
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train_scaled.shape[1],
                    kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        metrics=['accuracy', keras.metrics.Recall(name='recall'), 
                 keras.metrics.Precision(name='precision')]
    )

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.005, patience=20, 
                              verbose=0, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, 
                                 min_lr=1e-7, verbose=0)

    # Class weights
    class_weights = {0: 1, 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}

    # Training with progress bar
    with st.spinner('Training model... This may take a few minutes.'):
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
            class_weight=class_weights
        )
    
    # Save model and preprocessing objects
    model.save('churn_model.h5')
    joblib.dump(encoder, 'churn_encoder.joblib')
    joblib.dump(scaler, 'churn_scaler.joblib')
    
    st.sidebar.success("✅ Model trained and saved successfully!")
    return model, encoder, scaler

# Prediction function
def predict_churn(input_data, model, encoder, scaler, threshold=0.5):
    """Predict churn for input data"""
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    categorical_cols = ['Geography', 'Gender']
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                     'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    # Encoding categorical variables
    encoded_data = encoder.transform(input_df[categorical_cols])
    encoded_columns = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=input_df.index)
    
    # Keep numerical as-is first
    numerical_df = input_df[numerical_cols].copy()
    
    # Combine (before scaling!)
    processed_data = pd.concat([numerical_df, encoded_df], axis=1)
    
    # Ensure same column order as training
    expected_columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_Germany', 'Geography_Spain', 'Gender_Male'
    ]
    processed_data = processed_data[expected_columns]
    processed_scaled = scaler.transform(processed_data)
    
    # Predict
    prediction_proba = model.predict(processed_scaled, verbose=0)[0][0]
    prediction_binary = 1 if prediction_proba > threshold else 0
    
    return prediction_proba, prediction_binary


# Load or train model
model, encoder, scaler = load_or_train_model()

# Single prediction page
if app_mode == "Predict Single Customer":
    st.header("Predict Churn for a Single Customer")
    
    # input form
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            age = st.slider("Age", 18, 100, 42)
            tenure = st.slider("Tenure (years with bank)", 0, 10, 2)
            balance = st.number_input("Account Balance", 0.0, 300000.0, 0.0)
            
        with col2:
            products_number = st.slider("Number of Products", 1, 4, 1)
            credit_card = st.radio("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            active_member = st.radio("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0)
            country = st.selectbox("Country", ["France", "Germany", "Spain"])
            gender = st.radio("Gender", ["Male", "Female"])
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # input data
        input_data = {
            'CreditScore': credit_score,
            'Geography': country,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': products_number,
            'HasCrCard': credit_card,
            'IsActiveMember': active_member,
            'EstimatedSalary': estimated_salary
        }
        
        # Prediction
        with st.spinner('Making prediction...'):
            probability, prediction = predict_churn(input_data, model, encoder, scaler)
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Churn Probability", f"{probability:.2%}")
            st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")
        
        with col2:
            # Progress bar for probability
            st.write("Churn Risk Level:")
            if probability > 0.7:
                st.error(f"High Risk ({probability:.2%})")
            elif probability > 0.4:
                st.warning(f"Medium Risk ({probability:.2%})")
            else:
                st.success(f"Low Risk ({probability:.2%})")
            
            st.progress(float(probability))
        
        # Recommendations
        st.subheader("Recommendations")
        if prediction == 1:
            st.error("""
            ** Customer is at high risk of churning!**
            
            Recommended actions:
            - Contact customer for feedback
            - Offer personalized incentives
            - Assign to retention specialist
            - Review service quality
            """)
        else:
            st.success("""
            ** Customer is likely to stay**
            
            Maintenance actions:
            - Continue current service level
            - Monitor for changes in behavior
            - Consider upselling opportunities
            """)

# Batch prediction page
elif app_mode == "Batch Prediction":
    st.header("Batch Prediction from CSV File")
    
    uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                               'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            
            if all(col in batch_data.columns for col in required_columns):
                st.success(" File uploaded successfully!")
                st.dataframe(batch_data.head())
                
                if st.button("Predict Churn for All Customers"):
                    # Make predictions
                    predictions = []
                    probabilities = []
                    
                    with st.spinner('Processing batch predictions...'):
                        for _, row in batch_data.iterrows():
                            input_data = row.to_dict()
                            prob, pred = predict_churn(input_data, model, encoder, scaler)
                            predictions.append(pred)
                            probabilities.append(prob)
                    
                    # Add predictions to dataframe
                    results_df = batch_data.copy()
                    results_df['Churn_Probability'] = probabilities
                    results_df['Predicted_Churn'] = predictions
                    results_df['Churn_Risk'] = np.where(
                        results_df['Churn_Probability'] > 0.7, 'High',
                        np.where(results_df['Churn_Probability'] > 0.4, 'Medium', 'Low')
                    )
                    
                    # results
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)
                    
                    # Summary 
                    churn_count = results_df['Predicted_Churn'].sum()
                    total_customers = len(results_df)
                    churn_rate = churn_count / total_customers
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Customers", total_customers)
                    col2.metric("Predicted to Churn", churn_count)
                    col3.metric("Churn Rate", f"{churn_rate:.2%}")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f" Missing required columns. File must contain: {', '.join(required_columns)}")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Model training and evaluation page
elif app_mode == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    
    if st.button("Retrain Model"):
        with st.spinner('Training model...'):
            model, encoder, scaler = train_model()
        st.success("Model retrained successfully!")
    
    # Load test data 
    data = pd.read_csv("Churn_Modelling.csv")
    data.drop(columns=['CustomerId', 'Surname'], inplace=True)
    
    X = data.drop(columns=['Exited'])
    y = data['Exited']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocess test data
    X_test_encoded = encoder.transform(X_test[['Geography', 'Gender']])
    df_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(['Geography', 'Gender']))
    
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X_test_final = pd.concat([X_test[numerical_cols].reset_index(drop=True), df_test_encoded.reset_index(drop=True)], axis=1)
    
    X_test_scaled = scaler.transform(X_test_final)
    
    # Make predictions
    y_log = model.predict(X_test_scaled, verbose=0)
    y_pred = np.where(y_log > 0.5, 1, 0)
    
    # Display metrics
    st.subheader("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Stay', 'Churn'], 
           yticklabels=['Stay', 'Churn'],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    st.pyplot(fig)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**
- Predicts bank customer churn using deep learning
- Built with TensorFlow/Keras and Streamlit
- Uses customer demographic and financial data
""")