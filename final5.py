import unicodedata
import streamlit as st
import boto3
import json
from PIL import Image
from PyPDF2 import PdfReader
import io
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_mic_recorder import speech_to_text
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          )
from gtts import gTTS
from io import BytesIO
from IPython.display import Audio

# Constants and configuration
KB_ID = "GH4OHHIDTG"
MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
NUM_RESULTS = 10
REGION = "us-west-2"
bucket_name = 'energy-consumption-aws'

# AWS credentials
aws_access_key_id = 'AKIA3MJBUE3HR6EJWBHT'
aws_secret_access_key = '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'


API_url= "https://api-bee-system-cluster01.iems-acl.wise-insightapm.com"
portal_url= "https://portal-apm-system-cluster01.iems-acl.wise-insightapm.com"
User_Name= "iemsdemo.acl@wiseiot.com"
Password= "QGR2YW50ZWNoRGVtMA=="


# Initialize AWS clients
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=REGION,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

s3_client = boto3.client('s3', region_name=REGION, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Load authentication config
with open('./config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

if "page" not in st.session_state:
    st.session_state['page'] = "Login"

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

def set_page(page):
    st.session_state['page'] = page
    st.rerun()

def upload_to_s3(file_content, bucket, object_name):
    try:
        s3_client.put_object(Bucket=bucket, Key=object_name, Body=file_content)
        return True
    except Exception as e:
        print(f"上傳文件失敗: {e}")
        return False

def convert_json_to_csv(data):
    datapoints = data[0]['datapoints']
    df = pd.DataFrame(datapoints, columns=['energy_consumption', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['target'] = data[0]['target']
    df['resourceId'] = data[0]['resourceId']
    df['subitemCode'] = data[0]['subitemCode']
    df = df[['target', 'resourceId', 'subitemCode', 'energy_consumption', 'year', 'month', 'day', 'hour', 'minute']]
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def api_login():
    myobj = {"userName": User_Name, "password": Password}
    response = requests.post(portal_url + "/auth/api/v1/login", json=myobj)
    if response.status_code == 200:
        token = response.json().get("accessToken")
        return token
    else:
        st.error("API login failed")
        return None

def fetch_consumption_data(token, start_ts, end_ts, node_id):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
    request_body = {
        "timeoffset": "00:00:00",
        "timezone": "Asia/Hong_Kong",
        "range": {"from": start_ts, "to": end_ts},
        "language": "zh",
        "targets": [
            {
                "target": "energy",
                "queryType": "bems_multi",
                "apmOrgId": 1,
                "resourceId": node_id,
                "subitemCode": "01000",
                "formulaType": 183,
                "formulaUnit": "day"
            }
        ]
    }
    response = requests.post(API_url + "/v1/simplejson/query/new", headers=headers, data=json.dumps(request_body))
    if response.status_code == 200:
        data = response.json()
        csv_data = convert_json_to_csv(data)
        if upload_to_s3(csv_data, bucket_name, f"consumption_data_{node_id}_{start_ts}_{end_ts}.csv"):
            st.success(f"Data successfully uploaded to S3 as CSV: {bucket_name}/consumption_data_{node_id}_{start_ts}_{end_ts}.csv")
        else:
            st.error("Failed to upload data to S3")
        return data
    else:
        st.error("Failed to fetch API data")
        return None

def fetch_demand_minute_data(token, start_ts, end_ts, node_id):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
    request_body = {
        "sensors": [
            {
                "default": 0,
                "nodeId": node_id,
                "sensorName": "15 Minutes 電_AP",
                "sensorType": "ANOM"
            }
        ],
        "notAllowDefault": False,
        "retTsType": "unixTs",
        "startTs": start_ts,
        "endTs": end_ts
    }
    response = requests.post(portal_url + "/api-apm/api/v1/hist/raw/data", headers=headers, data=json.dumps(request_body))
    if response.status_code == 200:
        data = response.json()
        csv_data = convert_json_to_csv(data)
        if upload_to_s3(csv_data, bucket_name, f"demand_minute_data_{start_ts}_{end_ts}.csv"):
            st.success(f"Data successfully uploaded to S3 as CSV: {bucket_name}/demand_minute_data_{start_ts}_{end_ts}.csv")
        else:
            st.error("Failed to upload data to S3")
        return data
    else:
        st.error("Failed to fetch minute data")
        return None

def fetch_demand_span_data(token, start_ts, end_ts, node_id, interval):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
    request_body = {
        "sensors": [
            {
                "dataType": "max",
                "default": 0,
                "nodeId": node_id,
                "sensorName": "15 Minutes 電_AP",
                "sensorType": "ANOM"
            }
        ],
        "notAllowDefault": False,
        "retTsType": "unixTs",
        "interval": interval,
        "startTs": start_ts,
        "endTs": end_ts
    }
    response = requests.post(portal_url + "/api-apm/api/v1/hist/span/data", headers=headers, data=json.dumps(request_body))
    if response.status_code == 200:
        data = response.json()
        csv_data = convert_json_to_csv(data)
        if upload_to_s3(csv_data, bucket_name, f"demand_span_data_{start_ts}_{end_ts}_{interval}.csv"):
            st.success(f"Data successfully uploaded to S3 as CSV: {bucket_name}/demand_span_data_{start_ts}_{end_ts}_{interval}.csv")
        else:
            st.error("Failed to upload data to S3")
        return data
    else:
        st.error("Failed to fetch span data")
        return None

def call_claude_sonnet(search_results, question):
    prompt = f"""
        As a data analyst, you've been given a dataset and a graph based on energy consumption. You will conduct a detailed analysis of the current situation, covering the following aspects:
        Peaks and Troughs:
        Identify the highest and lowest energy consumption values in the dataset along with their corresponding dates.
        Overall Trend:
        Describe the overall trend in energy consumption, noting any significant periodic fluctuations or changes in trend.
        Statistical Insights:
        Calculate and provide the average, median, and standard deviation of energy consumption.
        Anomalies:
        Point out any notable high or low outliers, including the dates of these anomalies.
        Chart Interpretation:
        Utilize generated data charts and provide interpretations based on these charts.
        Here are the search results in numbered order:
        {search_results}
        given the data, answer the question: {question}
    """
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId=MODEL, accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")


# Building and floor mappings
building_mapping = {
    1024: "Sunny Building陽光大樓",
    1027: "Sunny Building陽光大樓",
    1025: "Sunny Building陽光大樓",
    1021: "Sunny Building陽光大樓",
    1026: "Sunny Building陽光大樓",
    1022: "Sunny Building陽光大樓",
    1023: "Sunny Building陽光大樓",
    1028: "Sunny Building陽光大樓"
}

floor_mapping = {
    1024: "1F Utility & Parking",
    1027: "1F Lab 實驗室",
    1025: "2F",
    1021: "3F",
    1026: "4F",
    1022: "5F",
    1023: "6F",
    1028: "7F"
}

def plot_retrieve_data(text_response):
    data = []
    print(text_response['retrievalResults'])
    # Process each retrieval result
    for result in text_response['retrievalResults']:
        # Split the content by carriage return to get individual lines
        lines = result['content']['text'].split('\r')
        # Split each line by comma and append to the data list
        for line in lines:
            split_line = line.split(',')
            data.append(split_line)

    print("data: ", data)
    if not data:
        st.error("No valid data found to plot.")
        return

    df = pd.DataFrame(data, columns=['type', 'id', 'subitemCode', 'building', 'floor', 'consumption', 'year', 'month', 'day', 'hour', 'minute'])
    df['consumption'] = pd.to_numeric(df['consumption'])
    df['year'] = pd.to_numeric(df['year'])
    df['month'] = pd.to_numeric(df['month'])
    df['day'] = pd.to_numeric(df['day'])
    df['hour'] = pd.to_numeric(df['hour'])
    df['minute'] = pd.to_numeric(df['minute'])

    average_consumption = df['consumption'].mean()
    median_consumption = df['consumption'].median()
    std_consumption = df['consumption'].std()
    max_consumption = df['consumption'].max()
    min_consumption = df['consumption'].min()
    max_consumption_date = df.loc[df['consumption'].idxmax()]
    min_consumption_date = df.loc[df['consumption'].idxmin()]

    # print(f'Average energy consumption: {average_consumption:.2f}')
    # print(f'Median energy consumption: {median_consumption:.2f}')
    # print(f'Standard deviation of energy consumption: {std_consumption:.2f}')
    # print(f'Highest energy consumption: {max_consumption:.2f} on {max_consumption_date["day"]}-{max_consumption_date["month"]}-{max_consumption_date["year"]}')
    # print(f'Lowest energy consumption: {min_consumption:.2f} on {min_consumption_date["day"]}-{min_consumption_date["month"]}-{min_consumption_date["year"]}')
    
    # Group by building and then by floor
    building_groups = df.groupby('building')
    for building, building_df in building_groups:
        plt.figure(figsize=(10, 6))
        floor_groups = building_df.groupby('floor')
        for floor, floor_df in floor_groups:
            plt.plot(floor_df['day'], floor_df['consumption'], marker='o', label=floor)
        plt.xlabel('Day')
        plt.ylabel('Energy Consumption')
        plt.title(f'Energy Consumption Over Time for {building}')
        plt.legend()
        plt.grid(True)
        plot_filename = f'energy_consumption_plot_{building}.png'
        plt.savefig(plot_filename)
        plt.close()
        st.image(plot_filename)

def call_backend_api(api_url, payload):
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return {"response": "抱歉，我無法回答你的問題。"}

def call_claude_sonnet2(search_results):
    prompt = f"""
       Extract Key Sections: Identify and extract key sections and headings to understand the structure of the document.
        Highlight Major Points: Summarize the main points from each section, focusing on the goals, techniques, and applications described.
        Concise Summary: Combine these points into a concise overview that covers the entire document.
        Here are the search results in numbered order:
        {search_results}
            """
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId=MODEL, accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")

def main():
    token = api_login()
    if "page" not in st.session_state:
        st.session_state['page'] = "Login"

    if st.session_state['page'] == "Login":
      # Creating a login widget
      try:
          authenticator.login()
      except LoginError as e:
          st.error(e)

      if st.session_state["authentication_status"]:
          authenticator.logout('Logout', 'sidebar')
          set_page("chatbot")
          st.write(f'歡迎來到 *{st.session_state["name"]}*')
          st.title('Some content')
      elif st.session_state["authentication_status"] is False:
          st.error('帳號／密碼錯誤')
      elif st.session_state["authentication_status"] is None:
          st.warning('請輸入你的帳號和密碼')
    
      # Arrange buttons in a single horizontal row with adjusted sizes and gaps
      col1, col2, col3 = st.columns(3, gap="small")
      with col1:
          if st.button("建立用戶"):
              set_page("register")
      with col2:
          if st.button("忘記帳號"):
              set_page("forget_username")
      with col3:
          if st.button("忘記密碼"):
              set_page("forget_password")

    elif st.session_state['page'] == "register":
        # Creating a new user registration widget
        try:
            (email_of_registered_user,
              username_of_registered_user,
              name_of_registered_user) = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                st.success('成功建立用戶')
                # set_page("chatbox")
        except RegisterError as e:
            st.error(e)

    elif st.session_state['page'] == "forget_password":
        # Creating a forgot password widget
        try:
            (username_of_forgotten_password,
            email_of_forgotten_password,
            new_random_password) = authenticator.forgot_password()
            if username_of_forgotten_password:
                st.success('New password sent securely')
                # Random password to be transferred to the user securely
            elif not username_of_forgotten_password:
                st.error('找不到帳號')
        except ForgotError as e:
            st.error(e)

    elif st.session_state['page'] == "forget_username":
        # Creating a forgot username widget
        try:
            (username_of_forgotten_username,
            email_of_forgotten_username) = authenticator.forgot_username()
            if username_of_forgotten_username:
                st.success('Username sent securely')
                # Username to be transferred to the user securely
            elif not username_of_forgotten_username:
                st.error('找不到電子郵件')
        except ForgotError as e:
            st.error(e)
    elif st.session_state['page'] == "chatbot":
        # def call_backend_api(api_url, payload):
        #     try:
        #         response = requests.post(api_url, json=payload)
        #         response.raise_for_status()
        #         return response.json()
        #     except requests.exceptions.RequestException as e:
        #         st.error(f"Error: {e}")
        #         return {"response": "抱歉，我無法回答你的問題。"}
        # Initialize chat sessions if not present
        if "chat_sessions" not in st.session_state:
            st.session_state["chat_sessions"] = {"Session 1": [{"role": "assistant", "content": "歡迎來到研華chatbot！我可以怎麼幫助你？"}]}
            st.session_state["current_session"] = "Session 1"

        # Function to create a new chat session
        def create_new_session():
            new_session_name = f"Session {len(st.session_state['chat_sessions']) + 1}"
            st.session_state["chat_sessions"][new_session_name] = [{"role": "assistant", "content": "歡迎來到研華chatbot！我可以怎麼幫助你？"}]
            st.session_state["current_session"] = new_session_name

        with st.sidebar:
            # Display username if available
            if "name" in st.session_state:
                st.markdown(f"## **{st.session_state['name']}**")
                # Adjust font size and make it bold
                st.markdown("<style>div.Widget.row-widget.stRadio button {font-size: 18px; font-weight: bold}</style>", unsafe_allow_html=True)

            # Chat session selection
            session_names = list(st.session_state["chat_sessions"].keys())
            selected_session = st.selectbox("選擇聊天室", session_names, index=session_names.index(st.session_state["current_session"]))
            if selected_session != st.session_state["current_session"]:
                st.session_state["current_session"] = selected_session

            # Button to create a new chat session
            if st.button("新聊天室"):
                create_new_session()

            # Display the first message from the user for each chat session
            st.write("### 聊天室紀錄")
            for session, messages in st.session_state["chat_sessions"].items():
                user_first_message = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
                if user_first_message:
                    st.write(f"{session}: {user_first_message}")

            # Add a logout button
            if st.button("登出"):
                authenticator.logout()
                set_page("Login")  # Redirect to login page after logout

        st.title("💬 研華Chatbot")
        st.caption("雲湧智生 — 智慧能源管理 ")

        for msg in st.session_state["chat_sessions"][st.session_state["current_session"]]:
            st.chat_message(msg["role"]).write(msg["content"])

        prompt = ""
        response1 = ""

        with st.container():
            prompt = speech_to_text(key='my_stt', start_prompt="語音輸入", stop_prompt="停止錄音")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("目前的用電有什麼異常"):
                    prompt = "目前的用電有什麼異常"
            with col2:
                if st.button("說一個笑話？"):
                    prompt = "說一個笑話？"
            col3, col4 = st.columns(2)
            with col3:
                if st.button("分析大樓的耗能資料"):
                    prompt = "分析大樓的耗能資料"
            with col4:
                uploaded_file = None
                uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
                if uploaded_file is not None:
                    # Read the uploaded PDF file
                    bytes_data = uploaded_file.getvalue()
                    # st.write("File uploaded successfully!")

                    # Display the PDF file content (optional)
                    with st.spinner("Processing..."):
                        # Open the PDF file and read its content
                        reader = PdfReader(io.BytesIO(bytes_data))
                        number_of_pages = len(reader.pages)
                        # st.write(f"Number of pages: {number_of_pages}")

                        # Extract and display text from each page (if needed)
                        pdf_text = ""
                        for page_num in range(number_of_pages):
                            page = reader.pages[page_num]
                            text = page.extract_text()
                            if text:
                                pdf_text += text + "\n\n"
                            pdf_text = pdf_text[:900]
                    with st.spinner('正在讀取文件······'):
                        response1 = call_claude_sonnet2(pdf_text)
        if response1:
            st.session_state["chat_sessions"][st.session_state["current_session"]].append({"role": "assistant", "content": response1})
            st.chat_message("assistant").write(response1)

            
        if prompt := st.chat_input() or prompt:
            st.session_state["chat_sessions"][st.session_state["current_session"]].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # payload = {"messages": st.session_state["chat_sessions"][st.session_state["current_session"]]}
            # backend_response = call_backend_api(backend_api_url, payload)
            # msg = backend_response.get("response", "抱歉，我無法回答你的提問。")
            # st.session_state["chat_sessions"][st.session_state["current_session"]].append({"role": "assistant", "content": msg})
            # st.chat_message("assistant").write(msg)

            text_response = bedrock_agent_runtime.retrieve(
                knowledgeBaseId=KB_ID,
                retrievalQuery={"text": prompt},
                retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 8}},
            )
            with st.spinner('正在生成資料······'):
                response = call_claude_sonnet(text_response, prompt)
            st.session_state["chat_sessions"][st.session_state["current_session"]].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    
            with st.spinner('正在生成音訊與圖表······'):
                #語音輸出
                sound_file = BytesIO()
                tts = gTTS(response, lang='zh', slow=False)
                tts.write_to_fp(sound_file)
                st.audio(sound_file)
            plot_retrieve_data(text_response)
            
            with st.popover("產生懶人包"): 
                with st.spinner('正在生成懶人包······'):   
                    text = call_claude_sonnet2(response)
                st.write(text)
            
def prediction(token, node_id):
    
    times = []
    values = []

    # Get the current date and the date 17 days ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=17)

    # Convert dates to timestamps in the required format
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    fetch_consumption_data(token, start_ts, end_ts, node_id)

    print("times: ", times, "values: ", values)
    
    request_body = {
        "times": times,
        "values": values,
        "model_type": "default",
        "data_frequency": "Day",
        "non_negative": True
    }

    # Call the prediction API
    api_url = "https://dataai-solution-practice-eks002.sa.wise-paas.com/prediction/integrated/"
    try:
        response = requests.post(api_url, json=request_body)
        print("response: ", response.json())
        if response.status_code == 200:
          prediction_data = response.json()
          return prediction_data
    except Exception as e:
        st.error(f"Failed to call prediction API: {e}")
        print(f"Failed to call prediction API: {e}")
        return False







if __name__ == "__main__":
    main()
