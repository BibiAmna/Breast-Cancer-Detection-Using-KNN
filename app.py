import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

# Load the model from the pickle file
with open('BreastCancerPrediction.pkl', 'rb') as file:
    model = pkl.load(file)

# Set the page configuration
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎀", layout="wide")

# Apply CSS for background image and pink/white color theme
st.markdown(
    """
    <style>
    .reportview-container {
        background: 'bg pink.jpg; /* Replace with your background image URL */
        background-size: cover;
    }
    .sidebar {
        background-color: #f8bbd0; /* Light Pink */
    }
    h1, h2, h3 {
        color: #d81b60; /* Dark Pink */
    }
    .stButton>button {
        background-color: #d81b60; /* Dark Pink */
        color: white;
    }
    .stSuccess {
        background-color: #c8e6c9; /* Light Green for positive predictions */
        color: black;
    }
    .stError {
        background-color: #ef5350; /* Red for errors */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for quitting the app
if 'quit' not in st.session_state:
    st.session_state.quit = False

# Check if the user wants to quit
if st.session_state.quit:
    st.write("The app has been exited. Close this tab.")
    st.stop()  # Stop further execution

# Streamlit title and description
st.title("Breast Cancer Prediction 🎀")

# Sidebar for navigation
st.sidebar.title("Information")
st.sidebar.subheader("Breast Cancer Information")
info_choice = st.sidebar.radio("Select an option:", ["Statistics", "Symptoms", "Diagnosis Methods"])

# Information content
if info_choice == "Statistics":
    st.sidebar.write("""
    ### Breast Cancer Statistics
    - Breast cancer is the most common cancer among women worldwide.
    - Approximately 1 in 8 women will be diagnosed with breast cancer in their lifetime.
    - Early detection can significantly increase the chances of successful treatment.
    """)

elif info_choice == "Symptoms":
    st.sidebar.write("""
    ### Common Symptoms of Breast Cancer
    - A lump or mass in the breast or underarm area.
    - Changes in the size, shape, or contour of the breast.
    - Unexplained swelling or irritation of the breast or nipple.
    - Discharge from the nipple that may be blood-stained or clear fluid.
    """)

elif info_choice == "Diagnosis Methods":
    st.sidebar.write("""
    ### Diagnosis Methods
    - **Mammogram**: An X-ray of the breast used to detect tumors.
    - **Ultrasound**: Uses sound waves to create images of the breast.
    - **Biopsy**: A sample of breast tissue is examined for cancer cells.
    - **MRI**: Magnetic Resonance Imaging for further evaluation of suspicious areas.
    """)

# Option to upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file containing patient data", type=["csv"])

# Process the uploaded CSV file if it exists
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    predicted = []

    # Ensure the required columns are present in the DataFrame
    required_columns = ['clump_thickness', 'uniformity_of_cell_size',
                        'uniformity_of_cell_shape', 'marginal_adhesion',
                        'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin',
                        'normal_nucleoli', 'mitosis']

    if all(col in df.columns for col in required_columns):
        # Prepare input data for predictions
        input_data = df[required_columns].to_numpy()

        # Make predictions
        predictions = model.predict(input_data)

        # Display the results
        st.subheader("Prediction Results")
        for i, prediction in enumerate(predictions):
            if prediction == 4:
                st.success(f"Patient {i + 1}: Positive for Cancer")
                predicted.append('Positive for Cancer')
            else:
                st.success(f"Patient {i + 1}: Negative for Cancer")
                predicted.append('Negative for Cancer')
        df['prediction'] = predicted

        # Add download button for CSV with predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction Results as CSV",
            data=csv,
            file_name='Prediction_Results.csv',
            mime='text/csv'
        )
    else:
        st.error("Uploaded CSV does not contain the required columns. Please upload a valid file.")

# Input field for a single array of values
input_values = st.text_input(
    "Or enter values as a comma-separated array:",
    placeholder="e.g., 5, 1, 1, 1, 2, 1, 3, 1, 1, 2"
)

# Convert input string to a list of values when the user presses the predict button
if st.button("Predict"):
    try:
        # Convert the input string to a numpy array
        input_list = [float(value) if i != 5 else int(value) for i, value in enumerate(input_values.split(','))]
        input_data = np.array([input_list])

        # Make prediction
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0] == 4:
            st.success("Prediction: Positive for Cancer")
        else:
            st.success("Prediction: Negative for Cancer")
    except ValueError:
        st.error("Invalid input format. Please ensure you enter the correct number of values.")
    except Exception as e:
        st.error(f"Error in input values: {e}")

# Quit button
if st.button("Quit"):
    st.session_state.quit = True
    st.write("The app has been exited. Close this tab.")
