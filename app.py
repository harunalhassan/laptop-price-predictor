#Importing Libraries
import streamlit as st
import pickle
import numpy as np
import time


# Set page configuration
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Center the title and add color using HTML and CSS
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            text-align: center;
            background-color: #36454F;
        }
        .stTitle {
            font-size: 3.5em;
            margin-bottom: 20px;
            animation: colorChange 5s infinite; /* Animation duration: 5 seconds, Infinite loop */
            color: blue; /* Initial color */
        }
        .stResult {
            padding: 10px;
            border: 3px solid green; /* Border color */
            display: inline-block;
            font-size: 2em;
            border-radius: 10px; /* Rounded corners */
        }
        @keyframes colorChange {
            0% { color: blue; } /* Blue */
            25% { color: green; } /* Green */
            50% { color: red; } /* Red */
            75% { color: purple; } /* Purple */
            100% { color: blue; } /* Back to Blue */
        }
    </style>
""", unsafe_allow_html=True)

# Display the title with the specified class
st.markdown("<h1 class='stTitle'>Laptop Price Predictor 💻</h1>", unsafe_allow_html=True)







#import model
# st.title("Laptop Price Predictor 💻")
pipe=pickle.load(open("pipe1.pkl","rb"))
df=pickle.load(open("df.pkl","rb"))

# making 3 cols left_column, middle_column, right_column
left_column, middle_column, right_column = st.columns(3)
with left_column:
    # brand input
    company = st.selectbox("Brand", df["Company"].unique())

with middle_column:
    # laptop type
    type = st.selectbox("Type", df["TypeName"].unique())

with right_column:
    # Ram size
    ram = st.selectbox("Ram (in GB)", df["Ram"].unique())

# making 3 cols left_column, middle_column, right_column
left_column, middle_column, right_column = st.columns(3)
with left_column:
    # Weight input
    weight = st.number_input("Weight of laptop in kg")

with middle_column:
    # Touchscreen
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

with right_column:
    # IPS display
    ips = st.selectbox("IPS Display", ["No", "Yes"])

# making 3 cols left_column, middle_column, right_column
left_column, middle_column, right_column = st.columns(3)
with left_column:
    # screen size
    Screen_size = st.number_input("Screen Size (in Inches)")

with middle_column:
    # resolution
  resolution = st.selectbox('Screen Resolution',['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600','2560x1440', '2304x1440'])
with right_column:
    # cpu input
    cpu = st.selectbox("CPU Brand", df["Cpu brand"].unique())

# making 3 cols left_column, middle_column, right_column
left_column,  right_column = st.columns(2)
with left_column:
    # hdd input
    hdd = st.selectbox("HDD(in GB)", [0, 128, 256, 512, 1024, 2048])


with right_column:
    # ssd input
    ssd = st.selectbox("SSD(in GB)", [0, 8, 128, 256, 512, 1024])

#gpu input
gpu=st.selectbox("GPU Brand",df["Gpu brand"].unique())

#os input
os=st.selectbox("OS Type",df["os"].unique())

if st.button("Pridict Price"):
    ppi = None
    if touchscreen=="Yes":
        touchscreen=1
    else:
        touchscreen=0

    if ips == "Yes":
        ips=1
    else:
        ips=0

    X_res=int(resolution.split("x")[0])
    Y_res=int(resolution.split('x')[1])
    ppi=((X_res ** 2)+(Y_res ** 2))**0.5/Screen_size
    query=np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query=query.reshape(1, 12)
   
    loader_container = st.empty()

        # Simulate a time-consuming prediction task
    with st.spinner(text='Predicting...'):
        time.sleep(1)  # Simulate a prediction time

    
    # st.title("The Predicted Price of Laptop = Rs "+str(int(np.exp(pipe.predict(query)[0]))))
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.markdown(f"<div class='stResult'>The Predicted Price of Laptop = Rs {predicted_price}</div>", unsafe_allow_html=True)


    st.balloons()
    st.snow()    


