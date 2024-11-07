import tempfile

import streamlit as st
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from loaders import *

st.set_page_config(layout="wide"
    , page_title="First Assistant",
    )

TIPOS_ARQUIVOS_VALIDOS = [
    'Chat','Analisador de Site', 'Analisador de Youtube', 'Analisador de Pdf', 'Analisador de CSV', 'Analisador de Texto', 'Analisador de Imagem'
]

openai = st.secrets["OPENAI_API_KEY"]
groq = st.secrets["GROQ_API_KEY"]

CONFIG_MODELOS = {'Groq': 
                        {'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
                         'chat': ChatGroq,
                         'api_key': groq},
                  'OpenAI': 
                        {'modelos': ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4o'],
                         'chat': ChatOpenAI,
                         'api_key': openai}}

MEMORIA = ConversationBufferMemory()

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Chat':
        documento = carrega_site(arquivo)
    if tipo_arquivo == 'Analisador de Site':
        documento = carrega_site(arquivo)
    if tipo_arquivo == 'Analisador de Youtube':
        documento = carrega_youtube(arquivo)
    if tipo_arquivo == 'Analisador de Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    if tipo_arquivo == 'Analisador de CSV':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    if tipo_arquivo == 'Analisador de Texto':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    
    return documento

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):

    documento = carrega_arquivos(tipo_arquivo, arquivo)
    system_message = '''Você é um assistente amigável chamado First Assistant.
    Você possui acesso às seguintes informações vindas 
    de um documento {}: 


    {}
    ####

    
Instruções de comportamento formatação e estilo:

        1. Use **negrito** para dar mais significado a palavras-chave.
        2. Utilize as informações fornecidas para basear as suas respostas.
        3. Utilize as informações fornecidas para basear as suas respostas. E retorne respostas completas, não apenas resumos.
        4. Use markdown para formatar sua resposta.
        5. Utilize cabeçalhos (##, ###) para organizar as informações em seções.
        6. Use listas com marcadores (-) ou numeradas (1., 2., 3.) para apresentar pontos importantes.
        7. Destaque palavras-chave ou frases importantes usando **negrito**.
        8. Utilize *itálico* para ênfase adicional quando apropriado.
        9. Se relevante, inclua citações usando o formato de bloco (>).
        10. Para informações técnicas ou códigos, use blocos de código com ``` .
        11. Crie tabelas quando apropriado para apresentar dados de forma organizada.
        12. Use emojis 🎯 ocasionalmente para adicionar um toque visual, mas não exagere.
        13. Conclua sua resposta com um breve resumo ou chamada para ação.
        14. Ocasionalmente, utilise cores em palavras chaves para deixar a resposta mais legível.
       
       Lembre-se:
        - Utilize as informações fornecidas no contexto para basear suas respostas.
        - Forneça respostas completas e detalhadas, não apenas resumos.
        - Sempre que houver $ na sua saída, substitua por S.
        - Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue", sugira ao usuário carregar novamente o First Assistant.
        - Quando for solicitado um código, responda com os blócos de código necessários. Se possível, forneça comentários 
        dentro do código para esclarecer o que está fazendo.
        - As respostas devem estar dentro de blocos de código Python QUANDO NECESSÁRIO e serem fáceis de entender.
        
        Ememplo de resposta bem formatada para códigos:

        ```python
        import pandas as pd

        # Carregar o DataFrame
        df = pd.read_csv('file.csv')
        ```

        Agora, por favor, responda à pergunta do usuário de forma detalhada, bem formatada e estilosa:
        '''.format(tipo_arquivo, documento)

     
        

    print(system_message)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain

def pagina_chat():
    st.header('_FIRST_ :violet[Assistant]', divider='violet')

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carrege o First Assistant antes de digitar')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o First Assistant')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages
            }))
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Analisador de Site':
            arquivo = st.text_input('Digite a url do site')
        if tipo_arquivo == 'Analisador de Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
        if tipo_arquivo == 'Analisador de Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
        if tipo_arquivo == 'Analisador de CSV':
            arquivo = st.file_uploader('Faça o upload do arquivo csv', type=['.csv'])
        if tipo_arquivo == 'Analisador de Texto':
            arquivo = st.file_uploader('Faça o upload do arquivo txt', type=['.txt'])
        if tipo_arquivo == 'Chat':
            arquivo = "https://firstbrazil.com.br/"
        if tipo_arquivo == 'Analisador de Imagem':
            arquivo = st.file_uploader('Faça o upload do arquivo png', type=['.png'])

    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelo', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = CONFIG_MODELOS[provedor]['api_key']
        
    if st.button('Inicializar o First Assistant', use_container_width=True):
            carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
            st.session_state['memoria'] = MEMORIA
    

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()
