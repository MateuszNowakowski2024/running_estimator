import streamlit as st
import os
import joblib
import pandas as pd
import json
import uuid
from datetime import datetime
from langfuse.decorators import observe
from langfuse.openai import OpenAI

#
#  BACKEND
#

# Retrieve secrets from environment variables (assumes they're set externally)
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

# Paths to local model & training data files
# (Update these if you want them to come from environment variables as well)
MODEL_PATH          = "halfmarathon_running_time_estimator_v5"
TRAINING_DATA_PATH  = "training_df.csv"

# Verify the essential environment variables are loaded
# (optional checks, remove or adjust as desired)
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in environment variables.")
if not LANGFUSE_SECRET_KEY:
    st.warning("LANGFUSE_SECRET_KEY not found in environment variables.")
if not LANGFUSE_PUBLIC_KEY:
    st.warning("LANGFUSE_PUBLIC_KEY not found in environment variables.")
if not LANGFUSE_HOST:
    st.warning("LANGFUSE_HOST not found in environment variables.")

@st.cache_resource
def load_model():
    # Check if the model file exists locally
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the file is in the same folder as app.py.")
        return None
    try:
        # Load the model from local path
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None
    return model

@st.cache_resource
def load_training_data():
    # Check if the training data file exists locally
    if not os.path.exists(TRAINING_DATA_PATH):
        st.error(f"Training data file not found at {TRAINING_DATA_PATH}. Please ensure the file is in the same folder as app.py.")
        return None
    try:
        # Load the training data from local path
        training_df = pd.read_csv(TRAINING_DATA_PATH, sep=',')
    except Exception as e:
        st.error(f"An error occurred while loading the training data: {e}")
        return None
    return training_df

# Load the model and stop the script if it fails
model = load_model()
if model is None:
    st.stop()

# Load the training data and stop the script if it fails
training_df = load_training_data()
if training_df is None:
    st.stop()

# Calculate medians from training data
medians = training_df.groupby(['płeć', 'kategoria_wiekowa_num']).median()

# BMI Tempo Offsets DataFrame
data = {
    "Age Group": [
        "20-30", "20-30", "20-30", "20-30", "20-30", "20-30", "20-30", "20-30",
        "30-40", "30-40", "30-40", "30-40", "30-40", "30-40", "30-40", "30-40",
        "40-50", "40-50", "40-50", "40-50", "40-50", "40-50", "40-50", "40-50",
        "50-60", "50-60", "50-60", "50-60", "50-60", "50-60", "50-60", "50-60",
        "60-70", "60-70", "60-70", "60-70", "60-70", "60-70", "60-70", "60-70",
        "70+",   "70+",   "70+",   "70+",   "70+",   "70+",   "70+",   "70+"
    ],
    "Gender": ["Male", "Male", "Male", "Male", 
               "Female", "Female", "Female", "Female"] * 6,
    "BMI Category": ["Underweight", "Normal Weight", "Overweight", "Obese"] * 12,
    "BMI Range": ["<18.5", "18.5-24.9", "25-29.9", ">30"] * 12,
    "Tempo Offset (sec/km)": [
         5,  0, 10, 20,  8,  0, 12, 25,
         7,  0, 12, 25, 10,  0, 15, 30,
        10,  0, 15, 30, 12,  0, 20, 35,
        15,  0, 20, 35, 18,  0, 25, 40,
        20,  0, 25, 45, 25,  0, 30, 50,
        30,  0, 35, 55, 35,  0, 40, 60
    ]
}

bmi_tempo_offsets_df = pd.DataFrame(data)

# Helper functions
def calculate_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi <= 24.9:
        return 'Normal Weight'
    elif 25 <= bmi <= 29.9:
        return 'Overweight'
    else:
        return 'Obese'

def prepare_input_data(age_category, gender, tempo_5km_seconds, bmi_category):
    gender_dict = {'Male': 0, 'Female': 1}
    gender_num = gender_dict.get(gender)

    # Get the tempo offset from the BMI dataframe
    tempo_offset = bmi_tempo_offsets_df[
        (bmi_tempo_offsets_df['Age Group'] == age_category) &
        (bmi_tempo_offsets_df['Gender'] == gender) &
        (bmi_tempo_offsets_df['BMI Category'] == bmi_category)
    ]['Tempo Offset (sec/km)'].values

    if len(tempo_offset) > 0:
        tempo_offset = tempo_offset[0]
    else:
        tempo_offset = 0

    # Multiply by 5 since offset is in sec/km for 5km
    adjusted_tempo_5km_seconds = tempo_5km_seconds + (tempo_offset * 5)

    age_dict = {
        '20-30': 20,
        '30-40': 30,
        '40-50': 40,
        '50-60': 50,
        '60-70': 60,
        '70+':    70
    }
    age = age_dict.get(age_category)

    return pd.DataFrame({
        'kategoria_wiekowa_num': [age],
        'płeć': [gender_num],
        '5_km_tempo_s': [adjusted_tempo_5km_seconds]
    })

def format_prediction(prediction):
    total_seconds = int(prediction[0])
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def save_conversation_locally():
    file_name = f'conversation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(file_name, 'w') as f:
        json.dump(st.session_state.messages, f, indent=2)
    with open(file_name, 'rb') as f:
        st.download_button(
            label="Download Conversation",
            data=f,
            file_name=file_name,
            mime="application/json",
            key="download_conversation"
        )

# Function to estimate time
def get_estimation(age_category, gender, tempo_5km, weight, height):
    # Check if tempo is within plausible range
    if tempo_5km < 3 or tempo_5km > 15:
        return "The tempo you provided seems implausible. Please ensure it is between 3 and 15 minutes per kilometer."

    # Convert tempo from minutes per km to seconds per km
    tempo_5km_seconds = tempo_5km * 60

    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    bmi_category = calculate_bmi_category(bmi)

    # Prepare input data
    input_data = prepare_input_data(age_category, gender, tempo_5km_seconds, bmi_category)

    # Fill other required columns
    required_columns = ['5_km_tempo_s', 'tempo_stabilność', 'płeć', 'kategoria_wiekowa_num']
    for col in required_columns:
        if col not in input_data.columns:
            if (input_data['płeć'][0], input_data['kategoria_wiekowa_num'][0]) in medians.index:
                input_data[col] = medians.loc[
                    (input_data['płeć'][0], input_data['kategoria_wiekowa_num'][0]), col
                ]
            else:
                input_data[col] = 0  # Default to 0 if no matching group exists

    # Predict
    prediction = model.predict(input_data)
    # Format prediction
    estimated_time = format_prediction(prediction)
    return estimated_time

# Define the function schema for OpenAI
functions = [
    {
        "name": "get_estimation",
        "description": "Estimate the half marathon running time based on user's data",
        "parameters": {
            "type": "object",
            "properties": {
                "age_category": {
                    "type": "string",
                    "description": "Age category of the user",
                    "enum": ["20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
                },
                "gender": {
                    "type": "string",
                    "description": "Gender of the user",
                    "enum": ["Male", "Female"]
                },
                "tempo_5km": {
                    "type": "number",
                    "description": "User's 5km tempo in minutes per km"
                },
                "weight": {
                    "type": "number",
                    "description": "User's weight in kilograms"
                },
                "height": {
                    "type": "number",
                    "description": "User's height in centimeters"
                }
            },
            "required": ["age_category", "gender", "tempo_5km", "weight", "height"]
        }
    }
]

system_prompt = """You are a running assistant that helps users estimate their half marathon running time. Collect the user's age category, gender, 5km tempo (in minutes per km), weight (in kg), and height (in cm). If the user doesn't know their tempo, help them calculate it by asking extra questions and providing more information.

If the user provides a tempo that is less than 3 minutes per km or more than 15 minutes per km, double-check with the user to ensure their answer is correct and in minutes per kilometer. If the user provides total time for running 5 km, help them calculate the tempo by dividing the total time by 5 and ensure the tempo is in minutes per kilometer.

When all the data is collected, calculate the BMI based on weight and height, provide the BMI result to the user then call the 'get_estimation' function with the collected data.
"""

@observe()
def handle_user_input(user_input, client, functions):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message('user').write(user_input)

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=st.session_state.messages,
        functions=functions,
        function_call="auto",
        temperature=0.0
    )

    assistant_message = response.choices[0].message

    # Handle function calls
    if assistant_message.function_call:
        function_name = assistant_message.function_call.name
        function_args_json = assistant_message.function_call.arguments
        function_args = json.loads(function_args_json)

        if function_name == 'get_estimation':
            estimated_time = get_estimation(
                age_category=function_args.get('age_category'),
                gender=function_args.get('gender'),
                tempo_5km=function_args.get('tempo_5km'),
                weight=function_args.get('weight'),
                height=function_args.get('height')
            )
            # Append the function response
            st.session_state.messages.append({
                "role": "function",
                "name": function_name,
                "content": estimated_time
            })

            # Send the updated messages back to the assistant
            response = client.chat.completions.create(
                model='gpt-4',
                messages=st.session_state.messages
            )
            final_reply = response.choices[0].message
            st.session_state.messages.append({
                "role": final_reply.role,
                "content": final_reply.content
            })
            st.chat_message('assistant').write(final_reply.content)
    else:
        # Normal message
        st.session_state.messages.append({
            "role": assistant_message.role,
            "content": assistant_message.content
        })
        st.chat_message('assistant').write(assistant_message.content)

#
# FRONT END
#

st.title('half marathon running time predictor :runner:')

st.markdown("<br><br>", unsafe_allow_html=True)

# Two modes for using the app
if 'mode' not in st.session_state:
    st.session_state['mode'] = None

col1, col2 = st.columns(2)
with col1:
    if st.button('AI ASSIST :mechanical_arm:', use_container_width=True):
        st.session_state['mode'] = 'ai_assist'
        st.rerun()
with col2:
    if st.button('NO AI ASSIST :muscle:', use_container_width=True):
        st.session_state['mode'] = 'no_ai_assist'
        st.rerun()

# Create the Langfuse/OpenAI client using your environment variables
# If you want to pass LANGFUSE keys or host, do so here (if relevant for your usage)
client = OpenAI(api_key=OPENAI_API_KEY)

if st.session_state['mode'] == 'ai_assist':
    # Initialize messages if not present
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "system", "content": system_prompt}]

    # Start the conversation if only system prompt is present
    if len(st.session_state.messages) == 1:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm here to help you estimate your half marathon running time. May I ask your age and gender?"
        })
        st.chat_message('assistant').write(st.session_state.messages[-1]['content'])
        st.rerun()

    # Display conversation history
    for message in st.session_state.messages[1:]:
        if message['role'] == 'user':
            st.chat_message('user').write(message['content'])
        elif message['role'] == 'assistant':
            st.chat_message('assistant').write(message['content'])
        elif message['role'] == 'function':
            # Optionally display function calls or function results
            pass

    # Buttons above chat input
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Reset Chat'):
            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            st.session_state['mode'] = None
            st.rerun()
    with col2:
        if st.button("Save Chat", key="save_chat_button"):
            save_conversation_locally()

    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        handle_user_input(user_input, client, functions)

elif st.session_state['mode'] == 'no_ai_assist':
    # Manual interface
    age_category = st.selectbox(
        'Select Your Age Category',
        options=['20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    )
    gender = st.selectbox('Select Your Gender', options=['Male', 'Female'])

    tempo_tooltip = (
        "To find your 5km tempo (pace), you can do a timed 5km run...\n"
        "Then, divide your total time (in minutes) by 5 to get your pace per km."
    )
    tempo_5km_input = st.number_input(
        'What Is Your 5km Tempo (minutes per km)',
        min_value=0.0,
        max_value=20.0,
        value=st.session_state.get('tempo_5km', 0.0),
        step=0.1,
        help=tempo_tooltip
    )
    tempo_5km = tempo_5km_input * 60  # Convert to seconds

    if st.button("I don't know my tempo, go with my age/gender average"):
        age_num = {'20-30': 20, '30-40': 30, '40-50': 40, '50-60': 50, '60-70': 60, '70+': 70}.get(age_category)
        gender_num = {'Male': 0, 'Female': 1}.get(gender)
        if (gender_num, age_num) in medians.index:
            st.session_state['tempo_5km'] = medians.loc[(gender_num, age_num), '5_km_tempo_s'] / 60
            st.rerun()

    bmi_tooltip = (
        "BMI (Body Mass Index)... formula: BMI = weight (kg) / (height (m)²)."
    )
    bmi = st.number_input(
        'Enter Your BMI (or calculate it)',
        min_value=0.0,
        max_value=50.0,
        value=st.session_state.get('bmi', 0.0),
        step=0.5,
        help=bmi_tooltip
    )

    # Two columns for BMI calculation
    bmi_col1, bmi_col2 = st.columns(2)
    with bmi_col1:
        if st.button('Help Me Calculate My BMI', use_container_width=True):
            st.session_state['bmi_button_clicked'] = True
            st.session_state['bmi_calculated'] = False
    with bmi_col2:
        if st.button("Skip (set as Normal)", use_container_width=True):
            st.session_state['bmi'] = 22.0
            st.rerun()

    if 'bmi_button_clicked' not in st.session_state:
        st.session_state['bmi_button_clicked'] = False
    if 'bmi_calculated' not in st.session_state:
        st.session_state['bmi_calculated'] = False

    if st.session_state['bmi_button_clicked'] and not st.session_state['bmi_calculated']:
        weight = st.number_input(
            'Enter Your Weight (kg)',
            min_value=0.0,
            max_value=300.0,
            value=0.0,
            step=1.0,
            key='weight_input'
        )
        height = st.number_input(
            'Enter Your Height (cm)',
            min_value=0.0,
            max_value=250.0,
            value=0.0,
            step=1.0,
            key='height_input'
        )

        if weight > 0 and height > 0:
            if st.button('Calculate BMI'):
                calculated_bmi = weight / ((height / 100) ** 2)
                st.session_state['bmi'] = calculated_bmi
                st.session_state['bmi_calculated'] = True
                st.session_state['bmi_button_clicked'] = False
                st.rerun()

    if st.session_state.get('bmi_calculated'):
        bmi = st.session_state['bmi']
        st.write(f'Your BMI is: {bmi:.2f}, which is considered {calculate_bmi_category(bmi)}')

    # Prepare model input
    bmi_category = calculate_bmi_category(bmi)
    input_data = prepare_input_data(age_category, gender, tempo_5km, bmi_category)
    required_columns = ['5_km_tempo_s', 'tempo_stabilność', 'płeć', 'kategoria_wiekowa_num']

    # Only enable the "Estimate" button if all key inputs are provided
    is_button_enabled = all([
        age_category,
        gender,
        tempo_5km > 0,
        bmi > 0
    ])

    estimate_button_col = st.columns([1, 8, 1])
    with estimate_button_col[1]:
        if st.button(
            'ESTIMATE MY HALF MARATHON RUNNING TIME',
            use_container_width=True,
            disabled=not is_button_enabled
        ):
            # Fill missing columns with median values if available, else 0
            for col in required_columns:
                if col not in input_data.columns:
                    if (input_data['płeć'][0], input_data['kategoria_wiekowa_num'][0]) in medians.index:
                        input_data[col] = medians.loc[
                            (input_data['płeć'][0], input_data['kategoria_wiekowa_num'][0]), col
                        ]
                    else:
                        input_data[col] = 0

            try:
                prediction = model.predict(input_data)
                estimated_time = format_prediction(prediction)
                st.markdown(f"### Estimated Half Marathon Running Time: **{estimated_time}**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

else:
    st.write('Please choose a mode to proceed.')
    st.image("runner.jpg")
