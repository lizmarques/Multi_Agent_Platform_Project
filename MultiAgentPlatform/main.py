import openai
import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
import os
import requests
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from langchain.agents import  create_csv_agent,create_pandas_dataframe_agent,initialize_agent, AgentType
from langchain.llms import AzureOpenAI

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish

from streamlit_chat import message
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from typing import Dict, List, Any
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.chains.base import Chain

from langchain.agents.agent_toolkits import GmailToolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain.utilities import BingSearchAPIWrapper
import matplotlib
import re
import speech_recognition as sr
import sounddevice as sd
import wavio as wv
from gtts import gTTS


# Conectando com o Azure OpenAI
# Fonte: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?tabs=command-line&pivots=programming-language-python
os.environ["OPENAI_API_KEY"] = "insira aqui a sua OPENAI API KEY"
os.environ["OPENAI_API_BASE"] =  "insira aqui a sua OPENAI API BASE"
os.environ["OPENAI_API_TYPE"] = "insira aqui a sua OPENAI API TYPE"
os.environ["OPENAI_API_VERSION"] = "insira aqui a sua OPENAI API VERSION"

# Bing
os.environ["BING_SUBSCRIPTION_KEY"] = "insira aqui a sua BING SUBSCRIPTION KEY"
os.environ["BING_SEARCH_URL"] = "insira aqui a sua BING SEARCH URL"

# Configurando a engine
llm=AzureOpenAI(model_kwargs={'engine':'gpt3'}, temperature=0)

# PandasAI
pandas_ai = PandasAI(llm,save_charts=True)

#Gmail toolkit
toolkit = GmailToolkit()


# Configurando o Bing tools
search = BingSearchAPIWrapper()

# Função
def bing_wrapper(input_text):
    search_results = search.run(f"site:dataside.com.br {input_text}")
    return search_results

#Tools
tools = [
    Tool(
        name = "Search Dataside",
        func=bing_wrapper,
        description="useful for when you need to answer questions about a consulting company called Dataside."
    )
]

# Configura o template base
template =  """Answer the following questions as best you can, but speaking as an expert. Provide detailed answers to questions ALWAYS based on the site and blog content of Dataside. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to provide comprehensive answers  ALWAYS based on the content of the site and blog of Dataside. The final answer should always be in portuguese (pt-br).

Question: {input}
{agent_scratchpad}"""

# Configura o prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        kwargs["agent_scratchpad"] = thoughts

        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Prompt
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

# Output Parser
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

# LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Tool names
tool_names = [tool.name for tool in tools]

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
)

# Configurações do Gmail agent
def write_email(draft):
    sent_draft = draft
    gmail_agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )
    answer2 = gmail_agent.run(f"Crie um rascunho no Gmail para mim que tem como principal objetivo: {sent_draft}. SEMPRE SIGA CADA UMA DAS orientações a seguir: Escreva utilizando a norma culta; Assine o email cordialmente como Liz Marques; Sob nenhuma circunstância você pode enviar a mensagem; Tudo deve ser SEMPRE escrito em português(pt-br); Ao final, escreva uma mensagem confirmando que a ação foi realizada com sucesso.")
    return answer2

# Função de gravação do áudio
def recording_audio():
    freq = 48000                                                    # Frequência
    duration = 12                                                    # Duração de cada gravação
    #st.info('Gravando...')
    recording = sd.rec(int(duration * freq),                        # Gravar matrizes NumPy contendo sinais de áudio
                       samplerate=freq, channels=2)
    sd.wait()                                                       # Verifica se a gravação já terminou
    wv.write("audio_vox.wav", recording, freq, sampwidth=2)         # Grava uma matriz Numpy em um arquivo WAV


# Função para converter o áudio em texto
def audio_to_text():
    # Iniciando o reconhecimento de fala
    r = sr.Recognizer()
    filename = "audio_vox.wav"

    # Abrindo o arquivo
    with sr.AudioFile(filename) as source:
        # "Escutando" o arquivo
        audio_data = r.listen(source)
        # Convertendo o audio em texto
        try:
            text = r.recognize_google(audio_data, language='pt-BR')         # Google Speech Recognition API
            return text.lower()

        except sr.UnknownValueError:
            return st.write("Tente gravar novamente.")


# Função que realiza a query
def query(file, query):
    agent = create_pandas_dataframe_agent(llm, file, verbose=True)
    answer = agent.run(query)
    return answer

# Insere o logo da Dataside
def dataside_logo():
    _, _, col3 = st.columns([1, 6, 1])
    with col3: st.image((Image.open(r"dataside_logo.png")), width=100)

def main():
    st.markdown("<h1 style='text-align: center; color: black;'>Multi Agent Platform<span>&#129302</span></h1>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["csv", "Website", 'Gmail'],
        icons=['filetype-csv', 'browser-chrome', 'envelope-at'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal")

    if selected == "csv":
        st.subheader("CSV Agent")
        st.markdown("O CSV Agent é um aplicativo que permite aos usuários extrair insights rapidamente "
                    "de seus arquivos .csv, sem a necessidade de conhecimentos em programação. Com uma interface "
                    "amigável, o agente CSV facilita a execução de análises com apenas alguns cliques, proporcionando "
                    "eficiência e praticidade para usuários de todos os níveis de habilidade técnica.")
        uploaded_file = st.file_uploader("Escolha um arquivo: ", type=['csv'])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df.head(3))
            user_query = st.text_input('Digite a sua pergunta: ')
            button = st.button("Enviar")

            if st.button:
                if user_query:
                    with st.spinner("Gerando as respostas..."):
                        st.write(pandas_ai.run(df,prompt=f"{user_query} Responda sempre em português(pt-br)"))

    if selected == "Website":
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Olá! Sou o SideAI, seu  Website Agent. Estou aqui para fornecer "
                                             "informações precisas e relevantes sobre a Dataside. Utilizando o motor "
                                             "de busca Bing, tenho acesso a todo o conteúdo disponível no site da empresa. "
                                             "Se você está procurando informações sobre os serviços, atualizações ou "
                                             "tem qualquer outra dúvida relacionada à Dataside, estou à disposição. "
                                             "Como eu posso te ajudar hoje? "]

        if 'past' not in st.session_state:
            st.session_state['past'] = ['Oi!']

        input_container = st.container()
        colored_header(label='', description='', color_name='blue-30')
        response_container = st.container()

        def get_text():
            input_text = st.text_input("Digite sua pergunta: ", "", key="input")
            return input_text

        def generate_response(prompt):
            response = zero_shot_agent.run(user_input)
            return response

        with input_container:
            user_input = get_text()

        with response_container:
            if user_input:
                response = generate_response(user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Felix")


    if selected == "Gmail":
        st.subheader("Gmail Agent")
        st.markdown("O Gmail Agent é uma aplicação que visa otimizar a escrita de e-mails no Gmail. Basta falar ou "
                    "escrever o conteúdo desejado, que o agente cria um rascunho em sua conta do Gmail. Posteriormente, "
                    "você pode revisar, editar e enviar sua mensagem com facilidade. Essa funcionalidade economiza "
                    "tempo e esforço, permitindo uma comunicação eficaz e focada nas mensagens.")

        gmail_email = st.text_area("O que você deseja enviar?")

        col1, _, col3 = st.columns([1, 6, 1])
        with col1:button = st.button("Enviar")
        with col3: button2 = st.button(":studio_microphone:")

        if button:
            if gmail_email:
                with st.spinner("Criando email..."):
                    st.write(write_email(gmail_email))
        if button2:
            st.info('Gravando...')
            recording_audio()
            gmail_email2 = audio_to_text()
            if gmail_email2:
                with st.spinner("Criando email..."):
                    st.write(write_email(gmail_email2))

if __name__ == '__main__':
    main()


