# import the packages
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from auto_ts import auto_timeseries
import xlrd

# set the web page configuration
st.set_page_config(page_title="Inventory Engine",
                   initial_sidebar_state="collapsed",
                   page_icon="😎")
# Define the title of the application & the markdown
st.title('Safety Stock & Re-order Level Calculator! ⚙️')
st.write('Generates accurate safety stock, reorder level in few simple steps!!')

# Define the tabs of the application
tabs = ["Application", "About"]
page = st.sidebar.radio("Pages 🧾", tabs)

# write the content for the page Application
if page == "Application":
    st.header('Upload Input Data 🔧 ')
    st.write('By default sample data is loaded. For your calculations please input new data.')
    st.subheader("1. Input demand history for minimum 12 periods.")
# write the collapse information
    with st.beta_expander("Input format"):
        st.write("Input should be a dataframe/table with two columns: Period and Demand. "
                 "Period can be in Day|Week|Month|Year "
                 "ideally in the format YYYY-MM-DD. "
                 "Demand column must be numeric.")
        demand_format = pd.DataFrame({'Period': ['2021-01-01','2021-02-01','2021-03-01','2021-04-01','2021-05-01','2021-06-01','2021-07-01','2021-08-01'], 'Demand': [10, 12, 13, 11, 12, 9,10,13]})
        st.write(demand_format)

    uploaded_file = st.file_uploader('')
    if uploaded_file is None:
        st.write("upload csv file")
    try:
        demand= pd.read_csv(uploaded_file)
    except:
        demand=demand_format
    demand= pd.DataFrame(demand)
    demand.columns= ['Period','Demand']
    st.subheader('2. Enter Supplier Lead Time 🔧 ')
    LeadTime = st.number_input('Caution:Demand & Lead time should be in same time frame. Example,If you have choosen weekly demand then the lead time should be in weeks.',1)
    st.subheader('3. Select Desired Service Level 🔧 ')
    ServiceLevel = st.slider('Service level is the probability fulfilling the expected demand with on hand inventory during the lead time', 0.90, 0.95, 0.99)
    LT= pd.DataFrame({'LeadTime':[LeadTime]})
    SL= ServiceLevel
# Subheader
    st.header('Recap Input Data')
    col1,col2, col3= st.beta_columns(3)
    col1.subheader("Demand Input")
    col1.write(demand)
    col2.subheader("Supplier Lead Time")
    col2.write(LT)
    col3.subheader("Service Level")
    col3.write(SL)
#Subheader
    demand['Period']= pd.to_datetime(demand['Period'])
    forecast_horizon=1
    model = auto_timeseries(forecast_period=forecast_horizon)
    model_fit = model.fit(demand, ts_column='Period', target='Demand')
    best_model=model.get_leaderboard()
    forecast_demand= np.average(model.predict(testdata=1,model='best',simple=True))
    Lead_Time_Demand = forecast_demand*LeadTime
    Standard_Deviation = demand['Demand'].std()
    Service_Factor = norm.ppf(SL)
    Lead_Time_Factor =np.sqrt(LeadTime)
    Safety_Stock =  Standard_Deviation*Service_Factor*Lead_Time_Factor
    Reorder_Point = Safety_Stock+Lead_Time_Demand
    st.header('Forecast Generated Using AutoML')
    st.write("Forecasting model used", best_model[0:1])
    st.header('We Are Done, Check The Result!')
    st.write('Safety Stock is', round(Safety_Stock,2))
    st.write('Reorder Point is', round(Reorder_Point,2))

if page == "About":
    st.image("Inv1.jpg")
    st.header("About")
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.write("Author:")
    st.markdown(""" **[Munikumar N.M](https://www.linkedin.com/in/munikumarnm/)**""")
    st.markdown("""**[Source code](https://github.com/Munikumarnm/streamlit)**""")
    st.write("Created on 20/03/2021")
    st.write("Last updated: **13/06/2021**")