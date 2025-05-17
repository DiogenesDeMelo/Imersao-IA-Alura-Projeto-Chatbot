import streamlit as st
import random

# Função para gerar um desafio aleatório
def gerar_desafio_aleatorio():
    desafios = [
        {
            "titulo": "Desafio da Caminhada",
            "descricao": "Caminhe 5 km por dia durante 7 dias.",
            "dificuldade": "Fácil",
            "pontos": 10,
            "duracao_dias": 7,
        },
        {
            "titulo": "Desafio da Alimentação Saudável",
            "descricao": "Coma 3 porções de frutas e verduras por dia durante 7 dias.",
            "dificuldade": "Médio",
            "pontos": 15,
            "duracao_dias": 7,
        },
        {
            "titulo": "Desafio da Leitura",
            "descricao": "Leia 30 minutos por dia durante 7 dias.",
            "dificuldade": "Médio",
            "pontos": 15,
            "duracao_dias": 7,
        },
        {
            "titulo": "Desafio do Sono",
            "descricao": "Durma 8 horas por noite durante 7 dias.",
            "dificuldade": "Difícil",
            "pontos": 20,
            "duracao_dias": 7,
        },
        {
            "titulo": "Desafio da Meditação",
            "descricao": "Medite 10 minutos por dia durante 7 dias.",
            "dificuldade": "Difícil",
            "pontos": 20,
            "duracao_dias": 7,
        },
    ]
    return random.choice(desafios)

# Função para adicionar um desafio à lista de desafios ativos do usuário
def aceitar_desafio(desafio):
    """
    Adiciona um desafio à lista de desafios ativos do usuário.

    Args:
        desafio (dict): Dicionário contendo informações do desafio
    """
    # Verificar se o desafio já está ativo
    if "desafios_ativos" not in st.session_state:
        st.session_state.desafios_ativos = []
    
    titulos_ativos = [d["titulo"] for d in st.session_state.desafios_ativos]
    if desafio["titulo"] not in titulos_ativos:
        st.session_state.desafios_ativos.append(desafio)
        st.success(f"🎯 Desafio aceito: {desafio['titulo']}")
        #adicionar_pontos(5, "Aceitou um novo desafio") #removi a chamada a função adicionar pontos, pois não existe.
    else:
        st.warning("Desafio já aceito")
    st.session_state.desafio_atual = {} #limpa o desafio atual para não exibir o mesmo desafio novamente

# Inicializa o estado da sessão para armazenar o desafio atual
if "desafio_atual" not in st.session_state:
    st.session_state.desafio_atual = {}

# Botão para gerar um novo desafio
if st.button("Gerar Novo Desafio", key="btn_novo_desafio"):
    desafio = gerar_desafio_aleatorio()
    st.session_state.desafio_atual = desafio  # Armazena o desafio no estado da sessão

# Exibe o desafio atual, se existir
if st.session_state.desafio_atual:
    desafio = st.session_state.desafio_atual
    st.markdown(f"""
    <div class="challenge-card">
        <h3>{desafio['titulo']}</h3>
        <p>{desafio['descricao']}</p>
        <p><strong>Dificuldade:</strong> {desafio['dificuldade']} | <strong>Pontos:</strong> {desafio['pontos']}</p>
        <p><strong>Duração:</strong> {desafio['duracao_dias']} dias</p>
    </div>
    """, unsafe_allow_html=True)

    # Botão para aceitar desafio
    if st.button("Aceitar Desafio", key="btn_aceitar"):
        aceitar_desafio(desafio)
        st.rerun()  # Força o Streamlit a atualizar a tela
