<h1 align="center"> Plataforma Multi Agente LangChain</h1>
<p align="center"> <img width="100px" heigth="300px" src="MultiAgentPlatform/imagens/agent_logo.png">
</p>

## [Vídeo] Demonstração da plataforma

https://github.com/lizmarques/Multi_Agent_Platform_Project/assets/90876339/dc103338-8dec-457e-947b-1cc38030be7e


## Links

- Aplicação no Streamlit - [App Folder](https://github.com/lizmarques/Multi_Agent_Platform_Project/tree/master/MultiAgentPlatform)

## Objetivo

O principal objetivo deste projeto foi explorar um pouco do framework LangChain e utilizar o Streamlit para criar uma mini plataforma com 3 agentes.

- CSV Agent: através deste agente, o usuário pode extrair insights rapidamente de arquivos .csv, sem a necessidade de conhecimentos em programação. Neste caso, acabei utilizando o PandasAI (https://github.com/gventuri/pandas-ai), no entanto, é possível utilizar o próprio agente csv do LangChain (https://python.langchain.com/docs/integrations/toolkits/csv).

- Website Agent: com a integração do Bing Search (https://python.langchain.com/docs/integrations/tools/bing_search), é possível utilizar todo o conteúdo de um website como base de conhecimento para um chatbot. Neste exemplo, eu direcionei a pesquisa exclusivamente para o site da Dataside, mas você pode direcionar para o site que quiser.

- Gmail Agent: utilizando o Gmail Toolkit (https://python.langchain.com/docs/integrations/toolkits/gmail), também oferecido pelo LangChain, criei um agente que tem como objetivo otimizar a escrita de e-mails. Nesta aplicação, implementei tanto um recurso por voz quanto por texto.


 ## Próximos Passos
- Permitir o upload de arquivos via Google Drive no CSV Agent
