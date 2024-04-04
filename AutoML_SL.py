# LangChain custom SAS Viya agent with memory and only two tools
# Resource: https://python.langchain.com/docs/modules/agents/how_to/custom_agent

import streamlit as st
import streamlit.components.v1 as components
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import random
import time
import urllib3
urllib3.disable_warnings()
import config
from urllib.parse import quote
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import base64
from langchain import hub
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain.agents import AgentExecutor
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.runnables import RunnableConfig
import saspy
        
## declare session state variables
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'access_token' not in st.session_state:
    st.session_state['access_token'] = ''
if 'show_login_form' not in st.session_state:
    st.session_state['show_login_form'] = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'response' not in st.session_state:
    st.session_state['response'] = None
if 'selected_question' not in st.session_state:
    st.session_state['selected_question'] = None
if 'sas' not in st.session_state:
    st.session_state['sas'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = None
if 'handler' not in st.session_state:
    st.session_state['handler'] = None

st.session_state["chat_history"] = []

sasserver = "https://viya4-s2.zeus.sashq-d.openstack.sas.com"
clientId='sas.ec'
clientSecret=''

## streamlit login form
def login_form():
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            return username, password
    return None, None

## Get User Token 
def getUserToken(username, password):
    try:
        loginCredentials={"grant_type":"password","username":username,"password":password}
        appBinding= clientId + ':' + clientSecret
        appBinding64= base64.b64encode(bytes(appBinding, 'utf-8'))
        url= sasserver+"/SASLogon/oauth/token"
        headers = {"Content-Type":"application/x-www-form-urlencoded", "Authorization":"Basic "}
        headers["Authorization"]= "Basic " + appBinding64.decode('ascii')
        response = requests.post(url, headers=headers, data=loginCredentials,verify=False)
        if response.status_code < 200 or response.status_code >= 300:
            logging.error("Error receiving user ("+username+") token :: "+response.text)
            return None, None
        access_token= response.json()['access_token']
        scope = response.json()['scope']
        print("access_token")
        return True, access_token
    except Exception as e:
        logging.warning("getUserToken :: {}".format(e))
        return False, None

# Load the Large Language Model - from Azure OpenAI
open_ai_key='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'
from langchain_openai import ChatOpenAI

## We'll use GPT-4 as deployed model
llm = ChatOpenAI(api_key=open_ai_key, model_name="gpt-3.5-turbo",temperature=0, max_tokens=2000)

# Define custom tools
## SASPY - see https://communities.sas.com/t5/SAS-Communities-Library/Submit-workloads-to-SAS-Viya-from-Python/ta-p/807263

from langchain.agents import tool, Tool

@tool

def generate_sas_code(prompt: str) -> int:
    """Returns the generated SAS code, based on the user prompt.
    Useful when people ask to generate SAS code that can be executed in a SAS environment or session.
    The response is returned as pure SAS code, executable in a SAS session.
    That means any explanation is commented out in SAS like fashion /* comment */"""
    import os
    from openai import OpenAI
    ## Setting Azure OpenAI environment variables
    # define client
    client = OpenAI(api_key=open_ai_key)
    #print("\nI will ask: ", prompt, '\n')
    response = client.chat.completions.create(
    model="gpt-3.5-turbo", # model = "deployment_name".
    messages=[
        {"role": "system", "content": """You are a SAS assistant that helps people write SAS code.
        The response you return has to be pure SAS code, executable in a SAS session.
        That means any explanation must be commented out in SAS like fashion. For example /* comment */ or * comment; .
        Explain your reasoning."""},
        {"role": "user", "content": "Return detailed table description for a SAS table CLASS in SASHELP library."},
        {"role": "assistant", "content": "PROC CONTENTS DATA=SASHELP.CLASS;run;quit;"},
        {"role": "user", "content": "Describe the SAS Data Set group from the health SAS library."},
        {"role": "assistant", "content": """"proc datasets library=health nolist; contents data=group (read=green) out=grpout;
            title  'The Contents of the GROUP Data Set';run;quit;"""},
        {"role": "user", "content": prompt},
            ]
        )
    sas_code = response.choices[0].message.content + '\n'
    return sas_code

def execute_sas_code(sas_code: str) -> str:
    """Useful when you need to execute generated SAS code.
        Input to this tool is correct SAS code, ready to be executed a SAS environment or session.
        Output comes in two parts:
        1. Results: the SAS code execution results comes at the top. The results are easy to identify. They can be found after the key word 'Results ---' Present the results in a clear, business-friendly way.
        2. Log: The SAS code execution log comes after the results. The results are easy to identify. They can be found after the key word 'Log --- '.
        Identify the results, summarize and provide a short explanation.
        If the execution result is not correct, an error message will be returned by the SAS code execution log.
        Error is signaled by the keyword 'ERROR'.
        If an error is returned, you may need to rewrite the SAS code, using the generate SAS code tool.
        The execute the SAS code.
        Provide a Final Result: Summarize and provide a short explanation of what was done and what the outcome was.
        If you encounter an issue detail what the issue is."""
    # SASPY auth and start session
   
    # Submit SAS code and parse results and the log
    sas_result = st.session_state['sas'].submit(sas_code, results='TEXT')
    result = json.dumps(sas_result['LST'])
    log = json.dumps(sas_result['LOG'])
    return ('Results --- \n', " ".join(result.split()), '---\n', 'Log --- \n', " ".join(log.split()), '---\n')

def getAutoMlTemplates() -> list:
    """
    Fetches and lists SAS Model Studio machine learning templates.

    This function is essential for identifying the available machine learning templates in SAS Model Studio. 
    It provides the necessary information, including template IDs, which are required when setting up an automated ML project using the createAutoMlProject() function.
    
    The function does NOT accept input parameters. 

    Returns:
    - List[dict]: A list of dictionaries, each representing a machine learning template. Each dictionary contains 'ID', 'Name', and 'Description' for the template.
    """
    headers = {'Authorization': 'bearer ' + st.session_state['access_token'] , 'Accept': 'application/json'}
    url=sasserver+'/mlPipelineAutomation/pipelineTemplates'
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        data = response.json()
        autoMlTemplates = [{"id": item["id"], "name": item["name"], "description": item["description"]}
                                for item in data["items"]
                            ]
        return autoMlTemplates
    except Exception as e:
        error_message = f"An error occurred: {e}"
        return [{"error": error_message}] 
       
def createAutoMlProject(projectName: str, templateId: str, trainingTable: str, targetVariable: str) -> dict:
    """
    Creates an automated machine learning pipeline in SAS Viya.

    This tool does NOT provide any information on machine learning pipeline templates. 

    This function simplifies the creation of a machine learning project by generating a pipeline based on the specified parameters. 
    It's particularly useful for quickly initiating projects within SAS Viya's modeling environment.

    Parameters:
    - projectName (str): Name of the project.
    - templateId (str): Identifier for the template to use. Obtain a suitable ID via getAutoMlTemplates() if unsure.
    - trainingTable (str): The dataset for training, in 'library.tableName' format (e.g., 'PUBLIC.HMEQ').
    - targetVariable (str): The variable the model aims to predict.

    Returns:
    - dict: A dictionary containing the URL to access the created project and the project's ID within SAS Viya's Model Studio.
      The dictionary includes the following keys:
      - 'model_studio_url' (str): A URL linking directly to the created project in SAS Viya's modeling application.
      - 'model_studio_projectId' (str): The unique identifier for the created project in SAS Viya's Model Studio.

    Example:
    createAutoMlProject("MyProject", "template123", "PUBLIC.HMEQ", "LoanDefault")
    """
    library=trainingTable.split('.')[0]
    table=trainingTable.split('.')[1]
    body={	"analyticsProjectAttributes": {
				"targetVariable": targetVariable,
				"partitionEnabled": True},
			"dataTableUri": '/dataTables/dataSources/cas~fs~cas-shared-default~fs~'+library+'/tables/'+table,
			"name": projectName,
			"settings": {
				"applyGlobalMetadata": True,
				"autoRun": True,
				"maxModelingTime": 100,
				"modelingMode": "Standard"},
			"type": "predictive",
			"pipelineBuildMethod": "template",
			"links": [{	"method": "GET",
						"rel": 'initialPipelineTemplate',
						"uri": '/mlPipelineAutomation/pipelineTemplates/'+templateId,
						"href": '/mlPipelineAutomation/pipelineTemplates/'+templateId,
						"type": 'application/octet-stream'}]
		}
    headers = {
				'Content-Type': 'application/vnd.sas.analytics.ml.pipeline.automation.project+json',
				'Accept': 'application/vnd.sas.analytics.ml.pipeline.automation.project+json',
				'Authorization': 'bearer ' + st.session_state['access_token'] 
			}
    url=sasserver+'/mlPipelineAutomation/projects'
    try:
        response=requests.post(url,headers=headers, data=json.dumps(body), verify=False)
        response.raise_for_status()
        data = response.json()
        model_studio_projectId=data["id"]
        ana_id=data["analyticsProjectAttributes"]["analyticsProjectId"]
        mstudio_url=sasserver+'/SASModelStudio/?projectUrl=/analyticsGateway/projects/'+ana_id+'&projectTabIndex=0'
        return {"model_studio_projectId": model_studio_projectId, "model_studio_url": mstudio_url}
    except Exception as e:
        error_message = f"An error occurred: {e}"
        return {"error": error_message}
    
def manageModel(model_studio_projectId: str, action: str) -> dict:
    """
    Manages the lifecycle actions (registration or publishing) of champion models within SAS Viya's automated machine learning pipelines. 

    This function requires a project ID, obtainable from the output of the createAutoMlProject() function, which initiates machine learning projects within SAS Viya's modeling environment. 
    
    This function checks the completion status of the specified project. If completed, it proceeds to either register the model for further use within SAS Viya's Model Manager or publish the model for operational use, deploying it to 'maslocal'.

    Parameters:
    - model_studio_projectId (str): The unique identifier for the machine learning project in SAS Viya's Model Studio, available as 'model_studio_projectId' from the output of createAutoMlProject().
    - action (str): The action to perform on the champion model. Valid options are 'register' or 'publish'.

    Returns:
    - dict: A dictionary containing the server's response upon the successful completion of the action, or an error message in case of failure. If the pipeline status is not 'completed', it returns the pipeline status.

    Example:
    # To register a model
    manageModel("12345678-abcd-1234-ef00-123456abcdef", "register")
    
    # To publish (and register) a model
    manageModel("12345678-abcd-1234-ef00-123456abcdef", "publish")
    """
	# First check the status of the machine learning pipeline
    headers = {
	    'Authorization': "bearer " + st.session_state['access_token'] ,
	    'Accept': "text/plain, application/vnd.sas.error+json"
	}
    url=sasserver+f"/mlPipelineAutomation/projects/{model_studio_projectId}/state"
    try:
        status_response=requests.get(url,headers=headers, verify=False)
        status_response.raise_for_status()  # Raises an HTTPError for bad responses
        pipeline_status = status_response.text
        print("Pipeline status:", pipeline_status)
    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching pipeline status: {str(e)}"
        print(error_message)
        return {"error": error_message}

    # Proceed with registration only if the status is 'completed'
    if pipeline_status.lower() == 'completed':
        # Adjust URL based on action <register> or <publish>
        if action == 'register':
            url = f"{sasserver}/mlPipelineAutomation/projects/{model_studio_projectId}/models/@championModel?action=register"
        elif action == 'publish':
            url = f"{sasserver}/mlPipelineAutomation/projects/{model_studio_projectId}/models/@championModel?action=publish&destinationName=maslocal"
        else:
            return {"error": "Invalid action specified. Only 'register' or 'publish' are allowed."}

        headers = {
		    'Content-Type': "application/json",
		    'Authorization': "bearer " + st.session_state['access_token'] ,
		    'Accept': "application/json, application/vnd.sas.analytics.ml.pipeline.automation.project.champion.model+json, application/vnd.sas.error+json"
		}
        payload=""
		#url=sasserver+f"/mlPipelineAutomation/projects/{model_studio_projectId}/models/@championModel?action=register"
        try:
            registration_response = requests.put(url,headers=headers, data=payload, verify=False)
            registration_response.raise_for_status()  # Raises an HTTPError for bad responses
            registration_result = registration_response.json()
            print(f"{action.capitalize()} response:", registration_result)
            return registration_result
        except requests.exceptions.RequestException as e:
            error_message =  f"{action.capitalize()} failed: {str(e)}"
            print(error_message)
            return {"error": error_message}
    else:
		# Return the pipeline status if not 'completed'
        message = f"Pipeline status is not completed: {pipeline_status}"
        print(message)
        return {"message": message}

from langchain.tools.base import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

class autoMlInput(BaseModel):
    projectName: str = Field(description="The name of the project")
    templateId: str = Field(description="The template Id of the project")
    trainingTable: str = Field(description="The name of the training table in this format: <libraryName.tableName>, e.g. public.hmeq")
    targetVariable: str = Field(description="The dependent variable of the dataset")

class manageModelInput(BaseModel):
    model_studio_projectId: str = Field(description="The unique identifier for the machine learning project in SAS Viya's Model Studio, obtainable as 'model_studio_projectId' from the output of the createAutoMlProject() function.")
    action: str = Field(description="The action to perform on the champion model. Valid options are 'register' or 'publish'.")

from typing import Union, Dict, Tuple
class autoMlTemplatesTool(BaseTool):
    name = "getAutoMlTemplates"
    description = """This tool fetches a comprehensive list of SAS Model Studio's machine learning templates, ideal for users looking to start automated ML projects or seeking an overview of available templates. 
        It outputs a collection, 'autoMlTemplates', comprising dictionaries that detail each template with its ID, Name, and Description. 
        This functionality is particularly valuable when initiating an AutoML project without a pre-specified template ID, allowing for the selection of the most fitting template based on the project's requirements.
        The function does not accept input parameters."""

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}

    def _run(self):
        ml_tempolates_list = getAutoMlTemplates()
        return ml_tempolates_list

    def _arun(self):
        raise NotImplementedError("getAutoMlTemplates does not support async")

# Describe the tools. The description is important for the LangChain chain.
tools = [
    Tool(
        name = "generate_sas_code",
        func = generate_sas_code,
        description = """
        Useful when people ask to generate SAS code that can be executed in a SAS environment
        or session.
        Input to this tool is the user prompt.
        Breakdown the user prompt.
        Explain your reasoning how you are going to construct the SAS code, from the user's prompt.
        Return the generated SAS code, based on the user's prompt or question.
        The response you return has to be pure SAS code, executable in a SAS session.
        That means any explanation must be commented out in SAS like fashion.
        For example /* comment */ or * comment ; .
        """
    ),
    Tool(
        name = "execute_sas_code",
        func = execute_sas_code,
        description = """Useful when you need to execute generated SAS code.
        Input to this tool is correct SAS code, ready to be executed.
        Output comes in two parts:
        1. Results: the SAS code execution results comes at the top. The results are easy to identify. They can be found between Results ---  and --- .
        2. Log: The SAS code execution log comes after the results. The results are easy to identify. They can be found between Log ---  and --- .
        Identify the results, summarize and provide a short explanation.
        Provide a Final Result: Summarize and provide a short explanation of what was done and what the outcome was.
        If you encounter an issue detail what the issue is."""
    ),
    autoMlTemplatesTool(),
    StructuredTool.from_function(
        name = "createAutoMlProject",
        func = createAutoMlProject,
        description = """This tool automates the creation of a machine learning pipeline in SAS Viya, tailored to your project's needs. 
        Simply provide a project name, select a template ID, specify your training data table, and the target variable you wish to predict. 
        If unsure about the template ID, use the autoMlTemplatesTool() tool to find the best match. 
        The function returns a dictionary containing a URL to access your project in the SAS Viya modeling application ('model_studio_url'), and the unique identifier of the created project ('model_studio_projectId'). 
        Please return the user with this model_studio_url and the model_studio_projectId in your final answer.
        Use this tool with arguments like "{"projectName":"testWes","templateId":"dm.basicbinarytargetpl.template", "trainingTable":"Public.HMEQ","targetVariable":"BAD"}" """,
        args_schema=autoMlInput
    ),
    StructuredTool.from_function(
        name = "manageModel",
        func = manageModel,
        description = """Manages the lifecycle actions (registration or publishing) of champion models within SAS Viya's automated machine learning pipelines. 
        This tool is designed to work with projects initiated by the createAutoMlProject Tool, from which the 'model_studio_projectId' can be obtained. 
        This tool first verifies the completion status of a specified machine learning project. If the project's pipeline status is 'completed', it proceeds to either register or publish the champion model based on the specified action. 
        Registration makes the model available for further use within SAS Viya's Model Manager.
        Publishing extends this by additionally deploying the model to a specific destination ('maslocal'), making it ready for inference or operational use.
        """,
        args_schema=manageModelInput
    )
        ]

# Create Prompt
## Now let us create the prompt. Because OpenAI Function Calling is finetuned for tool usage, we hardly need any instructions on how to reason, or how to output format.
## We will just have two input variables: input and agent_scratchpad. input should be a string containing the user objective.
## agent_scratchpad should be a sequence of messages that contains the previous agent tool invocations and the corresponding tool outputs.

# Add Memory

## Add a place for memory variables to go in the prompt
## Keep track of the chat history
## First, let’s add a place for memory in the prompt. We do this by adding a placeholder for messages with the key "chat_history". Notice that we put this ABOVE the new user input (to follow the conversation flow).

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class Data Engineer and Machine Learning Specialist, capable of writing and executing SAS code, creating automated machine learning pipelines, and fetching machine learning templates in SAS Model Studio.
            Please make sure you complete the objective above with the following rules:
            1/ Your job is to first breakdown the topic into relevant questions for understanding the topic in detail. You should have at max only 3 questions not more than that, in case there are more than 3 questions consider only the first 3 questions and ignore the others.
            2/ If asked to generate SAS code invoke the generate_sas_code tool.
            3/ If asked to run the generated code, invoke the execute_sas_code tool.
            Input to this tool is a detailed and correct SAS code, stripped of any plain text. If the query is not correct, an error message
            will be returned. If an error is returned, read the ERROR from the log. You may need to rewrite the SAS code, using the generate code tools,
            check the SAS code and execute the code again.
            4/ To create an automated machine learning pipeline in SAS Viya, invoke the createAutoMlProject function. 
            This is particularly useful for quickly setting up ML projects with specific objectives, training data, and target variables.
            Always include the model studio URL and the model studio projectId in tyour final answer to the user. 
            5/ If you need a list of available machine learning templates for SAS Model Studio, use the getAutoMlTemplates function. 
            This assists in selecting the appropriate template ID required for the createAutoMlProject function or for gaining an overview of possible ML project templates.
            Always include the complete list of dicts (with all keys and values) of templates in your final answer to the user.
            6/ If you are asked to register or publish a model then invoke the manageModel function. 
            This function requires the model studio projectId which is returned by the createAutoMlProject function.
            
            These tools and functions expand your capabilities to not only handle SAS code but also to streamline and enhance machine learning project initiation and template selection within SAS Viya."""
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Bind tools to LLM
## How does the agent know what tools it can use? In this case we’re relying on OpenAI function calling LLMs, which take functions as a separate argument and have been specifically trained to know when to invoke those functions.
## To pass in our tools to the agent, we just need to format them to the OpenAI function format and pass them to our model. (By bind-ing the functions, we’re making sure that they’re passed in each time the model is invoked.)

from langchain_community.tools.convert_to_openai import format_tool_to_openai_function # about to be deprecated

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# Create the Agent
## Putting those pieces together, we can now create the agent. We will import two last utility functions: a component for formatting intermediate steps (agent action, tool output pairs) to input messages that can be sent to the model, and a component for converting the output message into an agent action/agent finish.

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

## We can then set up a list to track the chat history

from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

from langchain.agents import AgentExecutor

st.session_state["handler"] = StdOutCallbackHandler()

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

with st.sidebar:
    if st.button("Connect to SASpy/SAS Viya" if not st.session_state['authenticated'] else "Disconnect from SAS Viya"):
        if st.session_state['authenticated']:
            st.session_state['authenticated'] = False
            st.session_state['access_token'] = None
            st.session_state['show_login_form'] = False
            st.session_state.messages = []
            st.success("Disconnected successfully.")
            st.experimental_rerun()
        else:
            st.session_state['show_login_form'] = True

    if st.session_state['show_login_form'] and not st.session_state['authenticated'] :
        username, password = login_form()
        
        if username and password:  # If credentials were provided
            sas = saspy.SASsession(cfgname="httpsviya")
            success, st.session_state['access_token'] = getUserToken(username, password)
            print(success)
            st.session_state['sas'] = sas
            lib_dict = sas.submit(
             """
             libname EUETS "/srv/nfs/kubedata/compute-landingzone";
             """
             )
            
            if sas != None: 
                st.session_state['authenticated'] = True
                st.session_state['show_login_form'] = False  
                st.success("Login successful!")
                st.experimental_rerun()
        
            else:
                st.error("Login failed. Please check your credentials.")
   
if st.session_state['authenticated']:
       
    if prompt := st.chat_input("Submit your question"):

        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
                container = st.container()
                st_callback = StreamlitCallbackHandler(container)
                print(st_callback)
                test = str(st_callback)
                print(test)
                cfg = RunnableConfig()
                cfg["callbacks"] = [st_callback]
                result = agent_executor.invoke({"input": prompt, "chat_history": st.session_state["chat_history"]}, cfg)
                st.session_state["chat_history"].extend(
                    [
                        HumanMessage(content=prompt),
                        AIMessage(content=result["output"]),
                    ]
                )
                
                