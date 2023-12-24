<h1 align="center"> Multi-Agent Platform using LangChain</h1>
<p align="center"> <img width="100px" heigth="300px" src="MultiAgentPlatform/imagens/agent_logo.png">
</p>

## [Video] Platform Demonstration

https://github.com/lizmarques/Multi_Agent_Platform_Project/assets/90876339/dc103338-8dec-457e-947b-1cc38030be7e


## Links

- Application in Streamlit - [App Folder](https://github.com/lizmarques/Multi_Agent_Platform_Project/tree/master/MultiAgentPlatform)

## Objective

The main objective of this project was to explore the LangChain framework and utilize Streamlit to create a mini-platform with 3 agents.

- CSV Agent: Through this agent, users can quickly extract insights from .csv files without the need for programming knowledge. In this case, I ended up using PandasAI (https://github.com/gventuri/pandas-ai); however, it's possible to use the LangChain csv agent itself (https://python.langchain.com/docs/integrations/toolkits/csv).

- Website Agent: With the integration of Bing Search (https://python.langchain.com/docs/integrations/tools/bing_search), it's possible to use the entire content of a website as knowledge for a chatbot. In this example, I directed the search exclusively to the Dataside website, but you can direct it to any website.

- Gmail Agent: Using the Gmail Toolkit (https://python.langchain.com/docs/integrations/toolkits/gmail), also offered by LangChain, I created an agent aimed at optimizing email writing. In this application, I implemented both voice and text input features.


 ## Next Steps
- Allow file uploads via Google Drive in the CSV Agent
