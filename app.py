import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

uploaded_files = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)
    df = pd.read_csv(BytesIO(bytes_data)) 
    st.write(df)
#    # Create a figure and axes
#     fig, ax = plt.subplots()

#     # Plot the data
#     ax.plot(df['Date'], df['Value'])

#     # Set labels and title
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_title('Simple Plot')

    # Display the plot in Streamlit
    # st.pyplot(df['Date'], df['Value'])
    st.scatter_chart(df[["Date","Value"]],x='Date',y='Value')


    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html