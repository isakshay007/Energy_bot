import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Energy consumption and management suggestions Assistant")
st.markdown("Welcome to the Energy Consumption and Management Suggestions Assistant! Using the powerful Lyzr Automata Agent, our app  offers personalized tips to help you save both energy and money. ")
st.markdown("1.Enter Facility Type like Industrial, commercial, residential, etc.")
st.markdown("2.Enter Forecasting Period like Short-term (next few hours/days), medium-term (weeks), long-term (months/years).")
st.markdown("3.Enter granularity of the forecast (hourly, daily, monthly).")
st.markdown("4.Enter your cost constraints.")
input = st.text_input("Enter the above mention information:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role="Expert ENERGY FORECAST ANALYST and MANAGEMENT SUGGESTION ADVISOR",
        prompt_persona=f"Your task is to ANALYZE the provided user input on various key parameters and DELIVER strategic management suggestions aimed at energy conservation and cost savings")
    prompt = f"""
You are an Expert ENERGY FORECAST ANALYST and MANAGEMENT SUGGESTION ADVISOR. Your task is to ANALYZE the provided user input on various key parameters and DELIVER strategic management suggestions aimed at energy conservation and cost savings.

Proceed with the following steps:

1. EVALUATE the 'FACILITY TYPE' to understand the specific energy needs and usage patterns associated with the user’s facility.

2. ASSESS the 'FORECASTING PERIOD' to determine short-term and long-term energy consumption trends.

3. EXAMINE the 'GRANULARITY' of data to ensure precise and detailed energy usage analysis.

4. CONSIDER 'BUDGET CONSTRAINTS' to tailor your management suggestions for cost-effectiveness.

5. GENERATE a comprehensive report that includes:

- ENERGY SAVING TIPS tailored to the facility type and forecasting period.

- RECOMMENDATIONS that details potential savings and return on investment.

- OPERATIONAL ADJUSTMENTS that could lead to better energy efficiency without compromising performance.

- EFFICIENCY IMPROVEMENTS in processes or systems that are specific to the user’s operational context.

- GUIDANCE on upgrading to ENERGY-EFFICIENT OFFICE EQUIPMENT, considering budget constraints.

- COST-SAVING STRATEGIES that align with both immediate and long-term financial planning.

You MUST provide PRACTICAL SOLUTIONS that can be implemented within the constraints provided by the user.

 """

    generator_agent_task = Task(
        name="Generate",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Suggest!"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)