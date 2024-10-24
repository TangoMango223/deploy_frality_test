"""
Main.py   Streamlit
"""

import streamlit as st
from typing import Dict, Any

# Set page config at the very beginning
st.set_page_config(page_title="StrideWell: Frailty Care Buddy", page_icon="🏥", layout="wide")

# Don't forget to add this import at the top of your file
import time

# Import statements
import os
from typing import Dict, Any

# LangChain Imports necessary for RAG
from langchain_openai import OpenAIEmbeddings # handle word embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import langchain_core.prompts.chat

# Import Pinecone
from pinecone import Pinecone


# Chain Extractors:
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# New imports
from langchain import hub

# New imports for creating document chains + retrieval
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Runnable PassThrough - node connections with no ops
from langchain_core.runnables import RunnablePassthrough

# Combine or stuffing chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Updated Color scheme for better visibility
COLOR_SCHEME = {
    'background': '#FFFFFF',
    'text': '#333333',
    'primary': '#00A86B',
    'secondary': '#F0F0F0',
    'accent': '#FF6B6B'
}

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {COLOR_SCHEME['background']};
        color: {COLOR_SCHEME['text']};
    }}
    .stButton>button {{
        color: white;
        background-color: {COLOR_SCHEME['primary']};
        border: none;
        border-radius: 20px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }}
    .stButton>button:hover {{
        background-color: {COLOR_SCHEME['accent']};
    }}
    .stTextInput>div>div>input, .stNumberInput>div>div>input {{
        color: {COLOR_SCHEME['text']};
        border-color: {COLOR_SCHEME['primary']};
        border-radius: 20px;
    }}
    .stSelectbox>div>div>select {{
        color: {COLOR_SCHEME['text']};
        border-color: {COLOR_SCHEME['primary']};
        border-radius: 20px;
    }}
    .stTab {{
        background-color: {COLOR_SCHEME['secondary']};
        color: {COLOR_SCHEME['text']};
        border-radius: 20px;
    }}
    h1, h2, h3 {{
        color: {COLOR_SCHEME['primary']};
    }}
    .stRadio > div {{
        background-color: {COLOR_SCHEME['secondary']};
        padding: 10px;
        border-radius: 20px;
    }}
    .stRadio > div > label {{
        color: {COLOR_SCHEME['text']};
        background-color: white;
        padding: 5px 10px;
        border-radius: 15px;
        margin-right: 10px;
    }}
    .stRadio > div > label:hover {{
        background-color: {COLOR_SCHEME['primary']};
        color: white;
    }}
    .story-container {{
        background-color: {COLOR_SCHEME['secondary']};
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
    }}
    .story-image {{
        width: 100%;
        border-radius: 10px;
    }}
    p, span, label, .stMarkdown {{
        color: {COLOR_SCHEME['text']};
    }}
</style>
"""

# Apply the custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Add navigation
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("""
    <div style="display: flex; justify-content: space-around;">
        <span>Caregiver Guides</span>
        <span>Caregiver Stories</span>
        <span>Ask the Community</span>
        <span>Resources</span>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.button("Sign In")

def generate_frailty_care_plan(
    first_gait_test_speed: float,
    first_gait_test_time: float,
    first_tug_test_time: float,
    gait_speed_test_risk: str,
    second_gait_test_speed: float,
    second_gait_test_time: float,
    second_tug_test_time: float,
    tug_test_risk: str,
    older_than_85: bool,
    is_male: bool,
    has_limiting_health_problems: bool,
    needs_regular_help: bool,
    has_homebound_health_problems: bool,
    has_close_help: bool,
    uses_mobility_aid: bool
):
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Initialize Pinecone
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

    # Connect to PineCone vector store
    vectorstore = PineconeVectorStore(
        index_name=st.secrets["INDEX_NAME"],
        embedding=embeddings
    )

    # Create the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Create the chat model
    chat = ChatOpenAI(temperature=0, model="gpt-4")

    # Create the prompts
    first_invocation_prompt = PromptTemplate.from_template("""
    You are an expert chatbot focused on frailty care, analyzing a patient's condition based on their PRISMA-7 survey responses and test results. Your task is to provide a factual analysis based solely on the given information. Do not make assumptions or infer information that is not explicitly stated.

    Patient's PRISMA-7 Responses and GAIT/TUG Test Results:
    <input>
    {input}
    </input>

    Relevant context from the knowledge base:
    <context>
    {context}
    </context>

    Please provide a detailed analysis considering only the information provided above. Address the following points:

    1. Frailty status: For each response from Patient's PRISMA-7 Responses and GAIT/TUG Test Results, think about how it contributes to the patient's frailty status.
    2. Overall frailty assessment: Summarize what the given responses and test results indicate about the patient's frailty status.
    3. Key areas of concern: Identify the most critical aspects that need addressing, based solely on the provided information.
    4. Potential risks: Discuss potential risks that are directly related to the information given.
    5. Care needs: Suggest interventions or support strategies that are relevant to the specific issues mentioned in the input.
    6. Interrelations: Explain how the different aspects of the patient's condition, as described in the input, may impact each other.

    In your analysis:
    - Be specific and refer only to the information provided in the input and context.
    - If the input or context doesn't provide sufficient information for any point, clearly state this lack of information.
    - Do not make assumptions or infer details that are not explicitly stated.
    - If you use information from the context, cite the source.

    Your goal is to provide an accurate understanding of the patient's frailty status based strictly on the given information. If there are gaps in the information or if more assessment is needed, state this clearly.

    Remember, do not provide any medical advice. Your role is to analyze the given information to support the development of a care plan by healthcare professionals.
    """)

    second_invocation_prompt = PromptTemplate.from_template("""
    You are an expert chatbot focused on frailty care, tasked with creating a comprehensive, personalized care plan. Your goal is to synthesize the provided analysis into an actionable, tailored care plan that supports both the caretaker and the frailty patient.

    You avoid humor or casual language due to the seriousness of the topic.

    You are provided the following information and analysis of the patient's condition.
    Patient's PRISMA-7 Responses, and Gait and TUG Test results:
    <input>
    {input}
    </input>

    I have conducted the following analysis of the patient's condition:
    <analysis>
    {analysis}
    </analysis>

    Based on this analysis, create a comprehensive care plan that addresses the specific needs and circumstances of this frailty patient. 

    You must begin your care plan by summarizing all the responses from the PRISMA-7 survey, and the Gait and TUG test results.
    Then, the care plan should:

    1. Provide a concise summary of the patient's overall frailty status, highlighting key areas of concern.

    2. Outline 4-5 key care recommendations. For each recommendation:
       a) Clearly state the recommendation
       b) Explain the rationale behind it, citing specific aspects of the patient's condition
       c) Provide detailed, practical steps for implementation
       d) Identify potential challenges and suggest strategies to overcome them

    3. Address safety considerations specific to this patient's situation, including both home safety and broader health and wellbeing measures.

    4. Suggest a monitoring and evaluation plan to track the patient's progress and adjust care as needed.

    5. Recommend specific resources or support services that would be particularly beneficial for this patient.

    6. Identify any areas where additional assessment or professional consultation might be necessary, explaining why.

    Throughout your care plan:
    - Ensure each recommendation is clearly linked to specific aspects of the patient's condition.
    - Prioritize interventions that address the most critical aspects of the patient's frailty status.
    - Consider the interplay between physical, cognitive, and social aspects of the patient's health.
    - Include both short-term interventions for immediate concerns and long-term strategies for ongoing care.
    - Provide clear, actionable guidance that can be readily implemented by caregivers.

    Your care plan should be comprehensive, practical, and tailored to both the patient's needs and the caretaker's ability to implement it.

    If there are any uncertainties or gaps in your knowledge, please say so and do not make up information. Clearly state what additional information or next steps would be required from healthcare providers.

    Your care plan should be comprehensive yet practical, providing clear guidance that can be readily implemented by caregivers while also serving as a valuable resource for healthcare professionals involved in the patient's care.

    Remember, your plan should be tailored to the patient's needs, and also meaningful to help caretakers as well.

    While knowledgeable about frailty care, you stay within your role of developing a care plan to support the caretaker and frailty patient, without providing definitive medical advice. Should there be any uncertainty, you should state this, and suggest the user to speak with a licensed healthcare professional.
    """)

    # Create the chains
    stuff_documents_chain = create_stuff_documents_chain(chat, first_invocation_prompt)
    qa = create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff_documents_chain)

    # Prepare the input data
    input_data = {
        "Are you older than 85 years?": "Yes" if older_than_85 else "No",
        "Are you male?": "Yes" if is_male else "No",
        "In general, do you have any health problems that require you to limit your activities?": "Yes" if has_limiting_health_problems else "No",
        "Do you need someone to help you on a regular basis?": "Yes" if needs_regular_help else "No",
        "In general, do you have any health problems that require you to stay at home?": "Yes" if has_homebound_health_problems else "No",
        "If you need help, can you count on someone close to you?": "Yes" if has_close_help else "No",
        "Do you regularly use a stick, walker or wheelchair to move about?": "Yes" if uses_mobility_aid else "No",
        "First Gait Test speed": f"{first_gait_test_speed} meters per second (m/s).",
        "First Gait Test time": f"{first_gait_test_time} seconds",
        "First TUG Test time": f"{first_tug_test_time} seconds",
        "Gait Speed Test Risk": gait_speed_test_risk,
        "Second Gait Test speed": f"{second_gait_test_speed} meters per second (m/s).",
        "Second Gait Test time": f"{second_gait_test_time} seconds",
        "Second TUG Test time": f"{second_tug_test_time} seconds",
        "TUG Test Risk": tug_test_risk,
    }

    # Run the first invocation
    first_result = qa.invoke(input={"input": str(input_data)})

    # Run the second invocation
    final_care_plan = chat.invoke(second_invocation_prompt.format(
        input=str(input_data),
        analysis=first_result["answer"]
    ))

    # Format the care plan as a big string
    care_plan_string = f"""
Care Plan:
{final_care_plan.content}

Sources used:
{chr(10).join(f"{i+1}. {source}" for i, source in enumerate(sorted(set(doc.metadata["source"] for doc in first_result["context"]))))}
"""

    return care_plan_string

def create_streamlit_ui() -> Dict[str, Any]:
    st.title("StrideWell: Your Frailty Care Buddy")

    # Create tabs for input and results
    tab1, tab2 = st.tabs(["Input Information", "Your Personalized Care Plan"])

    with tab1:
        st.header("PRISMA-7 Questionnaire")
        
        # Create two columns for the questionnaire
        col1, col2 = st.columns(2)
        
        with col1:
            older_than_85 = st.radio("Are you older than 85 years?", options=["Yes", "No"]) == "Yes"
            is_male = st.radio("Are you male?", options=["Yes", "No"]) == "Yes"
            has_limiting_health_problems = st.radio("In general, do you have any health problems that require you to limit your activities?", options=["Yes", "No"]) == "Yes"
            needs_regular_help = st.radio("Do you need someone to help you on a regular basis?", options=["Yes", "No"]) == "Yes"
        
        with col2:
            has_homebound_health_problems = st.radio("In general, do you have any health problems that require you to stay at home?", options=["Yes", "No"]) == "Yes"
            has_close_help = st.radio("If you need help, can you count on someone close to you?", options=["Yes", "No"]) == "Yes"
            uses_mobility_aid = st.radio("Do you regularly use a stick, walker or wheelchair to move about?", options=["Yes", "No"]) == "Yes"

        st.header("Gait and TUG Test Results")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("First Tests")
            first_gait_test_speed = st.number_input("First Gait Test speed (m/s)", min_value=0.0, step=0.1)
            first_gait_test_time = st.number_input("First Gait Test time (seconds)", min_value=0.0, step=0.1)
            first_tug_test_time = st.number_input("First TUG Test time (seconds)", min_value=0.0, step=0.1)
            gait_speed_test_risk = st.selectbox("Gait Speed Test Risk", ["Low", "Medium", "High"])

        with col2:
            st.subheader("Second Tests")
            second_gait_test_speed = st.number_input("Second Gait Test speed (m/s)", min_value=0.0, step=0.1)
            second_gait_test_time = st.number_input("Second Gait Test time (seconds)", min_value=0.0, step=0.1)
            second_tug_test_time = st.number_input("Second TUG Test time (seconds)", min_value=0.0, step=0.1)
            tug_test_risk = st.selectbox("TUG Test Risk", ["Low", "Medium", "High"])

        if st.button("Generate Care Plan"):
            # Store all input values in session state
            for key, value in locals().items():
                if key in ["older_than_85", "is_male", "has_limiting_health_problems", "needs_regular_help", 
                           "has_homebound_health_problems", "has_close_help", "uses_mobility_aid", 
                           "first_gait_test_speed", "first_gait_test_time", "first_tug_test_time", 
                           "gait_speed_test_risk", "second_gait_test_speed", "second_gait_test_time", 
                           "second_tug_test_time", "tug_test_risk"]:
                    st.session_state[key] = value
            
            # Set flag to generate care plan
            st.session_state.generate_clicked = True
            st.rerun()

    with tab2:
        if st.session_state.get('generate_clicked', False):
            with st.spinner("Generating your personalized care plan..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    # Update progress bar
                    progress_bar.progress(i + 1)
                    time.sleep(0.1)  # Simulate work being done

                care_plan = generate_frailty_care_plan(
                    first_gait_test_speed=st.session_state.first_gait_test_speed,
                    first_gait_test_time=st.session_state.first_gait_test_time,
                    first_tug_test_time=st.session_state.first_tug_test_time,
                    gait_speed_test_risk=st.session_state.gait_speed_test_risk,
                    second_gait_test_speed=st.session_state.second_gait_test_speed,
                    second_gait_test_time=st.session_state.second_gait_test_time,
                    second_tug_test_time=st.session_state.second_tug_test_time,
                    tug_test_risk=st.session_state.tug_test_risk,
                    older_than_85=st.session_state.older_than_85,
                    is_male=st.session_state.is_male,
                    has_limiting_health_problems=st.session_state.has_limiting_health_problems,
                    needs_regular_help=st.session_state.needs_regular_help,
                    has_homebound_health_problems=st.session_state.has_homebound_health_problems,
                    has_close_help=st.session_state.has_close_help,
                    uses_mobility_aid=st.session_state.uses_mobility_aid
                )
            st.session_state.care_plan_result = care_plan
            st.session_state.generate_clicked = False

        if 'care_plan_result' in st.session_state:
            st.header("Your Personalized Care Plan")
            st.markdown(f'<div style="background-color: {COLOR_SCHEME["secondary"]}; padding: 20px; border-radius: 20px;">{st.session_state.care_plan_result}</div>', unsafe_allow_html=True)
        elif st.session_state.get('generate_clicked', False):
            st.info("Generating your care plan. Please wait...")
        else:
            st.info("Please fill out the questionnaire and generate a care plan to see results.")

    return {
        "older_than_85": older_than_85,
        "is_male": is_male,
        "has_limiting_health_problems": has_limiting_health_problems,
        "needs_regular_help": needs_regular_help,
        "has_homebound_health_problems": has_homebound_health_problems,
        "has_close_help": has_close_help,
        "uses_mobility_aid": uses_mobility_aid,
        "first_gait_test_speed": first_gait_test_speed,
        "first_gait_test_time": first_gait_test_time,
        "first_tug_test_time": first_tug_test_time,
        "gait_speed_test_risk": gait_speed_test_risk,
        "second_gait_test_speed": second_gait_test_speed,
        "second_gait_test_time": second_gait_test_time,
        "second_tug_test_time": second_tug_test_time,
        "tug_test_risk": tug_test_risk,
    }

# Main execution
if 'generate_clicked' not in st.session_state:
    st.session_state.generate_clicked = False

user_inputs = create_streamlit_ui()

# If the generate button was clicked, switch to the second tab
if st.session_state.get('generate_clicked', False):
    st.query_params["tab"] = "Your Personalized Care Plan"
