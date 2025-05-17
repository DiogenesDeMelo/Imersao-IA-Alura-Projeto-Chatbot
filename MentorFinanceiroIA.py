"""
# Mentor Financeiro AI
# Aplicação de assistência financeira com interface Streamlit e recursos de gamificação
# Desenvolvido para fins educacionais
"""

# Importação das bibliotecas necessárias
import streamlit as st  # Biblioteca para criação de interface web
import google.generativeai as genai  # API do Google Generative AI (Gemini)
import os  # Para manipulação de variáveis de ambiente
import time  # Para pausas e simulação de processamento
import pandas as pd  # Para manipulação de dados
import matplotlib.pyplot as plt  # Para visualização de dados
import numpy as np  # Para operações numéricas
import random  # Para geração de desafios aleatórios
from datetime import datetime, timedelta  # Para manipulação de datas

# Configuração da página Streamlit
st.set_page_config(
    page_title="Mentor Financeiro AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para aplicar estilo CSS personalizado
def aplicar_estilo():
    """
    Aplica estilos CSS personalizados à interface Streamlit.
    Isso melhora a aparência visual e a experiência do usuário.
    """
    st.markdown("""
    <style>
        /* Estilo geral */
        .main {
            background-color: #5c1691;
            padding: 20px;
        }
        
        /* Estilo para cabeçalhos */
        h1, h2, h3 {
            color: #386641;
        }
        
        /* Estilo para caixas de informação */
        .info-box {
            background-color: #5c1691;
            border-left: 5px solid #81c784;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        /* Estilo para caixas de sucesso */
        .success-box {
            background-color: #d67322;
            border-left: 5px solid #4caf50;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        /* Estilo para caixas de alerta */
        .warning-box {
            background-color: #f5ea73;
            border-left: 5px solid #fbc02d;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        /* Estilo para barras de progresso */
        .stProgress > div > div {
            background-color: #080808;
        }
        
        /* Estilo para botões */
        .stButton button {
            background-color: #5c1691;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
        }
        
        .stButton button:hover {
            background-color: #386641;
        }
        
        /* Estilo para cartões de desafio */
        .challenge-card {
            background-color: #420773;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Estilo para medalhas e conquistas */
        .badge {
            display: inline-block;
            background-color: #81c784;
            color: white;
            border-radius: 20px;
            padding: 5px 15px;
            margin-right: 10px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Aplicar estilo personalizado
aplicar_estilo()

# --- Configuração da API Key do Google Generative AI ---
def configurar_api_key():
    """
    Configura a API Key do Google Generative AI.
    Verifica se a chave está disponível como variável de ambiente ou solicita ao usuário.
    
    Returns:
        bool: True se a configuração foi bem-sucedida, False caso contrário
    """
    # Verificar se já existe uma chave na sessão
    if 'api_key_configurada' in st.session_state and st.session_state.api_key_configurada:
        return True
    
    # Tentar obter a chave da variável de ambiente
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Se não encontrar, solicitar ao usuário
    if not api_key:
        st.sidebar.markdown("### Configuração da API Key")
        api_key = st.sidebar.text_input(
            "Insira sua Google API Key:",
            type="password",
            help="Obtenha sua chave em https://aistudio.google.com/app/apikey"
        )
        
        if not api_key:
            st.sidebar.warning("⚠️ API Key não fornecida. Algumas funcionalidades estarão limitadas.")
            return False
    
    # Tentar configurar a API com a chave fornecida
    try:
        genai.configure(api_key=api_key)
        st.session_state.api_key_configurada = True
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao configurar a API Key: {e}")
        st.session_state.api_key_configurada = False
        return False

# --- Configuração do Modelo Generativo ---
def configurar_modelo_gemini():
    """
    Configura o modelo Gemini com parâmetros específicos para geração de conteúdo.
    
    Returns:
        model: Instância configurada do modelo Gemini, ou None em caso de erro
    """
    if not configurar_api_key():
        return None
    
    # Configurações para geração de conteúdo
    generation_config = {
        "temperature": 0.75,  # Controla a criatividade (valores mais altos = mais criativo)
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,  # Aumentado para conselhos mais detalhados
    }
    
    # Configurações de segurança
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    try:
        # Inicializar o modelo Gemini
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        st.error(f"❌ Erro ao inicializar o modelo Gemini: {e}")
        return None

# --- Inicialização da Sessão ---
def inicializar_sessao():
    """
    Inicializa variáveis de sessão para armazenar dados do usuário e estado da aplicação.
    Isso permite persistência de dados entre interações com a interface.
    """
    # Variáveis para dados do usuário
    if 'nome_usuario' not in st.session_state:
        st.session_state.nome_usuario = ""
    
    if 'dados_financeiros' not in st.session_state:
        st.session_state.dados_financeiros = {
            'renda_mensal': None,
            'despesas_fixas': {},
            'despesas_variaveis': {},
            'dividas': {},
            'metas': {}
        }
    
    # Variáveis para gamificação
    if 'pontos' not in st.session_state:
        st.session_state.pontos = 0
    
    if 'nivel' not in st.session_state:
        st.session_state.nivel = 1
    
    if 'conquistas' not in st.session_state:
        st.session_state.conquistas = []
    
    if 'desafios_ativos' not in st.session_state:
        st.session_state.desafios_ativos = []
    
    if 'desafios_concluidos' not in st.session_state:
        st.session_state.desafios_concluidos = []
    
    # Variáveis para controle de navegação
    if 'pagina_atual' not in st.session_state:
        st.session_state.pagina_atual = "boas_vindas"
    
    if 'historico_consultas' not in st.session_state:
        st.session_state.historico_consultas = []
    
    # Variável para controle de diagnóstico
    if 'diagnostico_realizado' not in st.session_state:
        st.session_state.diagnostico_realizado = False

# --- Funções de Gamificação ---
def adicionar_pontos(quantidade, motivo=""):
    """
    Adiciona pontos ao usuário e verifica se houve evolução de nível.
    
    Args:
        quantidade (int): Quantidade de pontos a adicionar
        motivo (str): Motivo pelo qual os pontos foram adicionados
    """
    # Adicionar pontos
    st.session_state.pontos += quantidade
    
    # Verificar evolução de nível (a cada 100 pontos)
    nivel_anterior = st.session_state.nivel
    st.session_state.nivel = 1 + (st.session_state.pontos // 100)
    
    # Se houve evolução de nível, adicionar conquista
    if st.session_state.nivel > nivel_anterior:
        adicionar_conquista(f"Nível {st.session_state.nivel} Alcançado! 🏆")
        st.balloons()  # Efeito visual de celebração
    
    # Registrar no histórico se houver motivo
    if motivo:
        st.success(f"🎉 +{quantidade} pontos: {motivo}")

def adicionar_conquista(conquista):
    """
    Adiciona uma nova conquista à lista de conquistas do usuário.
    
    Args:
        conquista (str): Descrição da conquista obtida
    """
    if conquista not in st.session_state.conquistas:
        st.session_state.conquistas.append(conquista)
        st.success(f"🏆 Nova conquista desbloqueada: {conquista}")

def gerar_desafio_aleatorio():
    """
    Gera um desafio financeiro aleatório para o usuário.
    
    Returns:
        dict: Dicionário contendo informações do desafio
    """
    desafios = [
        {
            "titulo": "Semana Sem Delivery",
            "descricao": "Evite pedir comida por delivery por uma semana inteira.",
            "dificuldade": "Médio",
            "pontos": 30,
            "duracao_dias": 7
        },
        {
            "titulo": "Dia de Registro Total",
            "descricao": "Registre absolutamente todos os seus gastos por um dia inteiro, até os centavos.",
            "dificuldade": "Fácil",
            "pontos": 15,
            "duracao_dias": 1
        },
        {
            "titulo": "Economia de R$50",
            "descricao": "Encontre formas de economizar R$50 esta semana em gastos que você normalmente faria.",
            "dificuldade": "Médio",
            "pontos": 25,
            "duracao_dias": 7
        },
        {
            "titulo": "Pesquisa de Preços",
            "descricao": "Compare preços de 5 produtos que você compra regularmente em pelo menos 3 estabelecimentos diferentes.",
            "dificuldade": "Médio",
            "pontos": 20,
            "duracao_dias": 3
        },
        {
            "titulo": "Dia Sem Gastos",
            "descricao": "Passe um dia inteiro sem gastar absolutamente nada.",
            "dificuldade": "Difícil",
            "pontos": 40,
            "duracao_dias": 1
        }
    ]
    
    # Selecionar um desafio aleatório
    desafio = random.choice(desafios)
    
    # Adicionar data de início e fim
    data_inicio = datetime.now()
    data_fim = data_inicio + timedelta(days=desafio["duracao_dias"])
    
    desafio["data_inicio"] = data_inicio
    desafio["data_fim"] = data_fim
    desafio["concluido"] = False
    
    return desafio

def aceitar_desafio(desafio):
    """
    Adiciona um desafio à lista de desafios ativos do usuário.
    
    Args:
        desafio (dict): Dicionário contendo informações do desafio
    """
    # Verificar se o desafio já está ativo
    titulos_ativos = [d["titulo"] for d in st.session_state.desafios_ativos]
    if desafio["titulo"] not in titulos_ativos:
        st.session_state.desafios_ativos.append(desafio)
        st.success(f"🎯 Desafio aceito: {desafio['titulo']}")
        adicionar_pontos(5, "Aceitou um novo desafio")

def concluir_desafio(indice):
    """
    Marca um desafio como concluído e concede os pontos correspondentes.
    
    Args:
        indice (int): Índice do desafio na lista de desafios ativos
    """
    if 0 <= indice < len(st.session_state.desafios_ativos):
        desafio = st.session_state.desafios_ativos[indice]
        desafio["concluido"] = True
        
        # Mover para desafios concluídos
        st.session_state.desafios_concluidos.append(desafio)
        st.session_state.desafios_ativos.pop(indice)
        
        # Adicionar pontos e conquista
        adicionar_pontos(desafio["pontos"], f"Concluiu o desafio: {desafio['titulo']}")
        
        # Verificar conquistas especiais
        if len(st.session_state.desafios_concluidos) == 1:
            adicionar_conquista("Primeiro Desafio Concluído! 🌟")
        elif len(st.session_state.desafios_concluidos) == 5:
            adicionar_conquista("Desafiador Experiente: 5 Desafios Concluídos! 🔥")
        elif len(st.session_state.desafios_concluidos) == 10:
            adicionar_conquista("Mestre dos Desafios: 10 Desafios Concluídos! 🏅")

# --- Funções de Diagnóstico Financeiro ---
def calcular_saude_financeira():
    """
    Calcula indicadores de saúde financeira com base nos dados do usuário.
    
    Returns:
        dict: Dicionário contendo indicadores de saúde financeira
    """
    dados = st.session_state.dados_financeiros
    
    # Valores padrão
    resultado = {
        "comprometimento_renda": 0,
        "endividamento": 0,
        "reserva_emergencia": 0,
        "score": 0,
        "classificacao": "Não disponível"
    }
    
    # Verificar se há dados suficientes
    if not dados["renda_mensal"]:
        return resultado
    
    # Calcular total de despesas fixas
    total_despesas_fixas = sum(dados["despesas_fixas"].values()) if dados["despesas_fixas"] else 0
    
    # Calcular total de despesas variáveis
    total_despesas_variaveis = sum(dados["despesas_variaveis"].values()) if dados["despesas_variaveis"] else 0
    
    # Calcular total de parcelas de dívidas
    total_parcelas_dividas = 0
    for divida in dados["dividas"].values():
        if "parcela_mensal" in divida and divida["parcela_mensal"]:
            total_parcelas_dividas += divida["parcela_mensal"]
    
    # Calcular total de despesas
    total_despesas = total_despesas_fixas + total_despesas_variaveis + total_parcelas_dividas
    
    # Calcular comprometimento de renda
    if dados["renda_mensal"] > 0:
        resultado["comprometimento_renda"] = (total_despesas / dados["renda_mensal"]) * 100
    
    # Calcular endividamento
    total_dividas = sum([d.get("valor_total", 0) for d in dados["dividas"].values()])
    if dados["renda_mensal"] > 0:
        resultado["endividamento"] = (total_dividas / (dados["renda_mensal"] * 12)) * 100
    
    # Calcular reserva de emergência
    reserva = dados.get("reserva_emergencia", 0)
    if total_despesas > 0:
        resultado["reserva_emergencia"] = reserva / total_despesas if reserva else 0
    
    # Calcular score de saúde financeira (0-100)
    score = 100
    
    # Penalizar por alto comprometimento de renda
    if resultado["comprometimento_renda"] > 80:
        score -= 40
    elif resultado["comprometimento_renda"] > 60:
        score -= 25
    elif resultado["comprometimento_renda"] > 40:
        score -= 10
    
    # Penalizar por alto endividamento
    if resultado["endividamento"] > 50:
        score -= 30
    elif resultado["endividamento"] > 30:
        score -= 20
    elif resultado["endividamento"] > 15:
        score -= 10
    
    # Bonificar por reserva de emergência
    if resultado["reserva_emergencia"] >= 6:
        score += 20
    elif resultado["reserva_emergencia"] >= 3:
        score += 10
    elif resultado["reserva_emergencia"] < 1:
        score -= 20
    
    # Garantir que o score esteja entre 0 e 100
    resultado["score"] = max(0, min(100, score))
    
    # Classificar saúde financeira
    if resultado["score"] >= 80:
        resultado["classificacao"] = "Excelente"
    elif resultado["score"] >= 60:
        resultado["classificacao"] = "Boa"
    elif resultado["score"] >= 40:
        resultado["classificacao"] = "Regular"
    elif resultado["score"] >= 20:
        resultado["classificacao"] = "Preocupante"
    else:
        resultado["classificacao"] = "Crítica"
    
    return resultado

def gerar_grafico_despesas():
    """
    Gera um gráfico de pizza com a distribuição das despesas do usuário.
    
    Returns:
        fig: Figura do matplotlib com o gráfico gerado
    """
    dados = st.session_state.dados_financeiros
    
    # Combinar todas as despesas em um único dicionário
    todas_despesas = {}
    
    # Adicionar despesas fixas
    for categoria, valor in dados["despesas_fixas"].items():
        todas_despesas[f"Fixo: {categoria}"] = valor
    
    # Adicionar despesas variáveis
    for categoria, valor in dados["despesas_variaveis"].items():
        todas_despesas[f"Variável: {categoria}"] = valor
    
    # Adicionar parcelas de dívidas
    for nome_divida, info_divida in dados["dividas"].items():
        if "parcela_mensal" in info_divida and info_divida["parcela_mensal"]:
            todas_despesas[f"Dívida: {nome_divida}"] = info_divida["parcela_mensal"]
    
    # Verificar se há dados para gerar o gráfico
    if not todas_despesas:
        return None
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ordenar despesas por valor (do maior para o menor)
    despesas_ordenadas = dict(sorted(todas_despesas.items(), key=lambda x: x[1], reverse=True))
    
    # Definir cores para o gráfico
    cores = plt.cm.tab20.colors
    
    # Criar gráfico de pizza
    wedges, texts, autotexts = ax.pie(
        despesas_ordenadas.values(),
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=cores[:len(despesas_ordenadas)]
    )
    
    # Personalizar aparência dos textos
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Adicionar legenda
    ax.legend(
        wedges,
        despesas_ordenadas.keys(),
        title="Categorias",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    # Adicionar título
    ax.set_title("Distribuição de Despesas Mensais", fontsize=16, pad=20)
    
    # Ajustar layout
    plt.tight_layout()
    
    return fig

def calcular_tempo_quitacao_dividas():
    """
    Calcula o tempo estimado para quitação de cada dívida e o total.
    
    Returns:
        dict: Dicionário com informações sobre tempo de quitação
    """
    dados = st.session_state.dados_financeiros
    resultado = {
        "dividas": {},
        "tempo_total_meses": 0,
        "valor_total": 0,
        "juros_total": 0
    }
    
    # Verificar se há dívidas cadastradas
    if not dados["dividas"]:
        return resultado
    
    tempo_maximo = 0
    
    # Calcular para cada dívida
    for nome, divida in dados["dividas"].items():
        if "valor_total" in divida and "parcela_mensal" in divida and divida["parcela_mensal"] > 0:
            # Dados básicos da dívida
            valor_total = divida["valor_total"]
            parcela_mensal = divida["parcela_mensal"]
            taxa_juros_mensal = divida.get("taxa_juros_mensal", 0) / 100 if "taxa_juros_mensal" in divida else 0
            
            # Inicializar valores
            saldo_devedor = valor_total
            meses = 0
            total_pago = 0
            total_juros = 0
            
            # Simular pagamentos até quitar
            while saldo_devedor > 0 and meses < 1000:  # Limite de 1000 meses para evitar loop infinito
                meses += 1
                
                # Calcular juros do mês
                juros_mes = saldo_devedor * taxa_juros_mensal
                total_juros += juros_mes
                
                # Atualizar saldo devedor
                amortizacao = min(parcela_mensal, saldo_devedor + juros_mes)
                total_pago += amortizacao
                saldo_devedor = saldo_devedor + juros_mes - amortizacao
                
                # Se a parcela não cobre nem os juros, ajustar
                if saldo_devedor > valor_total and parcela_mensal <= juros_mes:
                    meses = float('inf')
                    break
            
            # Armazenar resultados
            resultado["dividas"][nome] = {
                "tempo_meses": meses,
                "total_pago": total_pago,
                "total_juros": total_juros
            }
            
            # Atualizar totais
            resultado["valor_total"] += valor_total
            resultado["juros_total"] += total_juros
            tempo_maximo = max(tempo_maximo, meses)
    
    # Definir tempo total como o maior tempo entre as dívidas
    resultado["tempo_total_meses"] = tempo_maximo
    
    return resultado

def sugerir_metodo_quitacao():
    """
    Sugere o melhor método de quitação de dívidas com base no perfil do usuário.
    
    Returns:
        dict: Dicionário com sugestão de método e explicação
    """
    dados = st.session_state.dados_financeiros
    
    # Verificar se há dívidas cadastradas
    if not dados["dividas"]:
        return {
            "metodo": "Nenhum",
            "explicacao": "Não há dívidas cadastradas para análise."
        }
    
    # Calcular indicadores
    total_dividas = sum([d.get("valor_total", 0) for d in dados["dividas"].values()])
    dividas_com_juros_altos = sum([1 for d in dados["dividas"].values() if d.get("taxa_juros_mensal", 0) > 5])
    quantidade_dividas = len(dados["dividas"])
    
    # Verificar perfil psicológico (simplificado)
    perfil_motivacional = "conquistas_rapidas"  # Padrão
    
    # Lógica de decisão
    if dividas_com_juros_altos > quantidade_dividas / 2:
        metodo = "Avalanche"
        explicacao = (
            "O método Avalanche consiste em pagar o mínimo em todas as dívidas e direcionar o valor extra para "
            "a dívida com a maior taxa de juros. Como você possui várias dívidas com juros altos, este método "
            "economizará mais dinheiro a longo prazo."
        )
    elif perfil_motivacional == "conquistas_rapidas" and quantidade_dividas > 2:
        metodo = "Bola de Neve"
        explicacao = (
            "O método Bola de Neve consiste em pagar o mínimo em todas as dívidas e direcionar o valor extra para "
            "a dívida com o menor saldo devedor. Este método proporciona vitórias rápidas que aumentam sua motivação, "
            "o que é ideal para seu perfil."
        )
    else:
        metodo = "Híbrido"
        explicacao = (
            "Um método híbrido é recomendado para seu caso. Comece quitando uma dívida pequena para ganhar motivação, "
            "depois foque nas dívidas com juros mais altos para economizar dinheiro a longo prazo."
        )
    
    return {
        "metodo": metodo,
        "explicacao": explicacao
    }

# --- Funções de Conteúdo Educacional ---
def obter_explicacao_termo_financeiro(termo):
    """
    Obtém explicação sobre um termo financeiro usando o modelo Gemini.
    
    Args:
        termo (str): Termo financeiro a ser explicado
        
    Returns:
        str: Explicação do termo financeiro
    """
    modelo = configurar_modelo_gemini()
    if not modelo:
        return "Não foi possível obter a explicação. Verifique a configuração da API Key."
    
    try:
        prompt = [
            f"Explique o termo financeiro '{termo}' de forma simples e didática, como se estivesse explicando para um adolescente.",
            "A explicação deve ter no máximo 3 parágrafos, usar linguagem acessível e incluir um exemplo prático do dia a dia.",
            "Responda em português do Brasil."
        ]
        
        response = modelo.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro ao obter explicação: {e}"

def gerar_dica_financeira_personalizada():
    """
    Gera uma dica financeira personalizada com base nos dados do usuário usando o modelo Gemini.
    
    Returns:
        str: Dica financeira personalizada
    """
    modelo = configurar_modelo_gemini()
    if not modelo:
        return "Não foi possível gerar uma dica personalizada. Verifique a configuração da API Key."
    
    dados = st.session_state.dados_financeiros
    saude = calcular_saude_financeira()
    
    try:
        # Construir prompt com dados do usuário
        prompt_parts = [
            f"Gere uma dica financeira personalizada com base nos seguintes dados:",
            f"- Comprometimento de renda: {saude['comprometimento_renda']:.1f}%",
            f"- Nível de endividamento: {saude['endividamento']:.1f}%",
            f"- Classificação da saúde financeira: {saude['classificacao']}",
        ]
        
        # Adicionar informações sobre dívidas, se disponíveis
        if dados["dividas"]:
            prompt_parts.append("- Possui dívidas ativas")
            
            # Adicionar tipos de dívidas
            tipos_dividas = [nome for nome in dados["dividas"].keys()]
            prompt_parts.append(f"- Tipos de dívidas: {', '.join(tipos_dividas)}")
        else:
            prompt_parts.append("- Não possui dívidas ativas")
        
        # Adicionar informações sobre reserva de emergência
        if saude["reserva_emergencia"] > 0:
            prompt_parts.append(f"- Possui reserva de emergência para {saude['reserva_emergencia']:.1f} meses")
        else:
            prompt_parts.append("- Não possui reserva de emergência")
        
        # Instruções para o formato da dica
        prompt_parts.extend([
            "A dica deve ser:",
            "1. Específica para a situação financeira descrita",
            "2. Prática e acionável (algo que a pessoa possa implementar imediatamente)",
            "3. Motivadora e positiva",
            "4. Curta (máximo de 3 frases)",
            "Responda em português do Brasil."
        ])
        
        response = modelo.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"Erro ao gerar dica personalizada: {e}"

def gerar_conselho_financeiro(preocupacao):
    """
    Gera um conselho financeiro personalizado com base na preocupação do usuário.
    
    Args:
        preocupacao (str): Preocupação financeira do usuário
        
    Returns:
        str: Conselho financeiro personalizado
    """
    modelo = configurar_modelo_gemini()
    if not modelo:
        return "Não foi possível gerar um conselho. Verifique a configuração da API Key."
    
    dados = st.session_state.dados_financeiros
    nome = st.session_state.nome_usuario
    
    try:
        # Construir prompt para o modelo
        prompt_parts = [
            f"Meu nome é {nome}.",
            f"Minha principal preocupação financeira é: '{preocupacao}'.",
        ]
        
        # Adicionar informações financeiras, se disponíveis
        if dados["renda_mensal"]:
            prompt_parts.append(f"Minha renda mensal aproximada é de R${dados['renda_mensal']:.2f}.")
        
        # Adicionar informações sobre dívidas, se disponíveis
        for nome_divida, info_divida in dados["dividas"].items():
            if "valor_total" in info_divida:
                prompt_parts.append(f"Tenho uma dívida de {nome_divida} no valor de R${info_divida['valor_total']:.2f}.")
        
        # Instruções para o formato do conselho
        prompt_parts.extend([
            "\nVocê é um consultor financeiro experiente, empático, motivador e bem detalhista.",
            "Preciso de ajuda para lidar com essa situação.",
            "Forneça para mim, em português do Brasil:",
            "1. Uma mensagem curta de encorajamento e validação dos meus sentimentos (1-2 frases).",
            "2. Um Planejamento Financeira detalhado, com base na minha renda, despesa e dívidas, de forma a traçar um plano objetivo e alcançável.",
            "3. Esse planejamento deverá conter valores, que demonstre como e quando eu posso atingir o objetivo de melhorar a saúde financeira.",
            "4. Caso verifique que com a minha renda atual não seja possível alcançar o objetivo, sugerir opções de renda extra para que a renda seja maximizada e então, conseguir quitar dívidas, ou metas financeiras.",
            "5. Uma dica extra ou uma reflexão positiva curta (1 frase).",
            "Seja claro, direto e use uma linguagem acessível. Evite jargões financeiros complexos."
        ])
        
        # Gerar resposta
        response = modelo.generate_content(prompt_parts)
        
        # Registrar consulta no histórico
        st.session_state.historico_consultas.append({
            "data": datetime.now(),
            "preocupacao": preocupacao,
            "conselho": response.text
        })
        
        # Adicionar pontos pela consulta
        adicionar_pontos(10, "Solicitou um conselho financeiro")
        
        return response.text
    except Exception as e:
        return f"Erro ao gerar conselho: {e}"

def simular_negociacao_divida(credor, valor_divida, dias_atraso):
    """
    Simula uma negociação de dívida usando o modelo Gemini.
    
    Args:
        credor (str): Nome do credor
        valor_divida (float): Valor da dívida
        dias_atraso (int): Dias de atraso
        
    Returns:
        str: Simulação de negociação
    """
    modelo = configurar_modelo_gemini()
    if not modelo:
        return "Não foi possível simular a negociação. Verifique a configuração da API Key."
    
    nome = st.session_state.nome_usuario
    
    try:
        # Construir prompt para o modelo
        prompt_parts = [
            f"Simule uma conversa de negociação de dívida entre {nome} e um atendente do(a) {credor}.",
            f"Valor da dívida: R${valor_divida:.2f}",
            f"Dias de atraso: {dias_atraso}",
            "\nA simulação deve incluir:",
            "1. Saudação inicial do atendente",
            "2. Como o cliente (eu) deve se apresentar e explicar a situação",
            "3. Perguntas que o atendente provavelmente fará",
            "4. Argumentos que posso usar para negociar um desconto ou parcelamento",
            "5. Possíveis propostas do atendente",
            "6. Como devo responder a cada proposta",
            "7. Conclusão da negociação",
            "\nFormate como um diálogo realista, com falas alternadas entre o atendente e o cliente.",
            "Use linguagem natural e realista para ambas as partes.",
            "Inclua dicas entre parênteses para me orientar durante a negociação.",
            "Responda em português do Brasil."
        ]
        
        # Gerar resposta
        response = modelo.generate_content(prompt_parts)
        
        # Adicionar pontos pela simulação
        adicionar_pontos(15, "Realizou uma simulação de negociação")
        
        return response.text
    except Exception as e:
        return f"Erro ao simular negociação: {e}"

# --- Componentes da Interface ---
def exibir_cabecalho():
    """
    Exibe o cabeçalho da aplicação com título e informações do usuário.
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🌟 Mentor Financeiro AI")
        st.markdown("### Seu assistente para trilhar o caminho da saúde financeira!")
    
    with col2:
        if st.session_state.nome_usuario:
            st.markdown(f"### Olá, {st.session_state.nome_usuario}!")
            st.markdown(f"**Nível:** {st.session_state.nivel}")
            st.markdown(f"**Pontos:** {st.session_state.pontos}")
            
            # Exibir medalha de acordo com o nível
            if st.session_state.nivel >= 5:
                st.markdown("🏆 **Especialista Financeiro**")
            elif st.session_state.nivel >= 3:
                st.markdown("🥈 **Estrategista Financeiro**")
            elif st.session_state.nivel >= 1:
                st.markdown("🥉 **Aprendiz Financeiro**")

def exibir_barra_lateral():
    """
    Exibe a barra lateral com menu de navegação e informações.
    """
    st.sidebar.title("Menu")
    
    # Verificar se o usuário já forneceu o nome
    if st.session_state.nome_usuario:
        # Botões de navegação
        if st.sidebar.button("📊 Dashboard", use_container_width=True):
            st.session_state.pagina_atual = "dashboard"
        
        if st.sidebar.button("💬 Consultor Virtual", use_container_width=True):
            st.session_state.pagina_atual = "consultor"
        
        if st.sidebar.button("📝 Diagnóstico Financeiro", use_container_width=True):
            st.session_state.pagina_atual = "diagnostico"
        
        if st.sidebar.button("🎯 Desafios", use_container_width=True):
            st.session_state.pagina_atual = "desafios"
        
        if st.sidebar.button("📚 Conteúdo Educacional", use_container_width=True):
            st.session_state.pagina_atual = "educacional"
        
        if st.sidebar.button("🏆 Conquistas", use_container_width=True):
            st.session_state.pagina_atual = "conquistas"
        
        # Separador
        st.sidebar.markdown("---")
        
        # Informações do usuário
        st.sidebar.markdown("### Seu Progresso")
        
        # Barra de progresso para o próximo nível
        progresso_nivel = (st.session_state.pontos % 100) / 100
        st.sidebar.progress(progresso_nivel, text=f"Progresso para Nível {st.session_state.nivel + 1}")
        
        # Exibir conquistas recentes
        if st.session_state.conquistas:
            st.sidebar.markdown("### Conquistas Recentes")
            for i, conquista in enumerate(st.session_state.conquistas[-3:]):
                st.sidebar.markdown(f"- {conquista}")
            
            if len(st.session_state.conquistas) > 3:
                st.sidebar.markdown(f"*...e mais {len(st.session_state.conquistas) - 3} conquistas*")
    
    # Configuração da API
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configurações")
    
    # Status da API
    if 'api_key_configurada' in st.session_state and st.session_state.api_key_configurada:
        st.sidebar.success("✅ API Gemini configurada")
    else:
        st.sidebar.warning("⚠️ API Gemini não configurada")
        configurar_api_key()
    
    # Informações sobre o aplicativo
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sobre")
    st.sidebar.info(
        "Mentor Financeiro AI\n\n"
        "Desenvolvido para fins educacionais\n\n"
        "Utiliza a API Gemini do Google para gerar conselhos financeiros personalizados."
        "O Conteúdo aqui gerado utiliza-se de Inteligência Artificial, que pode cometer erros. Sempre procure ajuda profissional especializada!"
    )

# --- Páginas da Aplicação ---
def pagina_boas_vindas():
    """
    Exibe a página de boas-vindas e coleta informações iniciais do usuário.
    """
    st.markdown("## 👋 Bem-vindo ao Mentor Financeiro!")
    
    st.markdown("""
    <div class="info-box">
        <h3>O que é o Mentor Financeiro AI?</h3>
        <p>Eu fui desenvolvido como um assistente inteligente que utiliza IA para ajudar você a melhorar sua saúde financeira, 
        oferecendo planejamentos financeiros personalizados, diagnósticos financeiros e desafios para desenvolver 
        hábitos financeiros saudáveis.</p>
        <p>Minha missão é ajudar pessoas em estado de endividamento a darem os primeiros passos para um vida financeira leve e tranquila!</p>        
    </div>
    """, unsafe_allow_html=True)
    
    # Coletar nome do usuário
    nome = st.text_input("Para começarmos, qual é o seu nome?", key="input_nome")
    
    if st.button("Começar Jornada", key="btn_comecar"):
        if nome:
            st.session_state.nome_usuario = nome
            st.session_state.pagina_atual = "dashboard"
            
            # Adicionar primeira conquista
            adicionar_conquista("Início da Jornada Financeira! 🚀")
            adicionar_pontos(10, "Iniciou sua jornada financeira")
            
            st.success(f"Olá, {nome}! Bem-vindo à sua jornada financeira!")
            st.balloons()  # Efeito visual de celebração
            
            # Recarregar a página para atualizar a interface
            st.rerun()
        else:
            st.error("Por favor, informe seu nome para continuar.")
    
    # Exibir recursos disponíveis
    st.markdown("### Recursos disponíveis:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Consultor Virtual Inteligente**: Tire suas dúvidas financeiras
        - **Diagnóstico Financeiro**: Avalie sua saúde financeira
        - **Planos de Ação Personalizados**: Receba orientações específicas
        """)
    
    with col2:
        st.markdown("""
        - **Desafios Financeiros**: Desenvolva hábitos saudáveis
        - **Conteúdo Educacional**: Aprenda sobre finanças
        - **Sistema de Gamificação**: Ganhe pontos e conquistas
        """)

def pagina_dashboard():
    """
    Exibe o dashboard principal com resumo da situação financeira do usuário.
    """
    st.markdown("## 📊 Dashboard")
    
    # Verificar se há dados financeiros
    if not st.session_state.dados_financeiros["renda_mensal"]:
        st.warning("Você ainda não completou seu diagnóstico financeiro. Complete-o para visualizar seu dashboard completo.")
        
        if st.button("Ir para Diagnóstico Financeiro"):
            st.session_state.pagina_atual = "diagnostico"
            st.rerun()
        
        # Exibir conteúdo limitado
        st.markdown("### Dica do Dia")
        st.info(
            "💡 **Comece anotando todos os seus gastos!**\n\n"
            "O primeiro passo para melhorar sua saúde financeira é entender para onde seu dinheiro está indo. "
            "Anote todos os seus gastos por pelo menos uma semana para identificar padrões."
        )
        
        return
    
    # Calcular indicadores de saúde financeira
    saude = calcular_saude_financeira()
    
    # Exibir resumo da saúde financeira
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Comprometimento de Renda",
            value=f"{saude['comprometimento_renda']:.1f}%",
            delta="-0.5%" if saude['comprometimento_renda'] < 80 else "0%"
        )
        
        if saude['comprometimento_renda'] > 80:
            st.error("⚠️ Acima do recomendado (80%)")
        elif saude['comprometimento_renda'] > 60:
            st.warning("⚠️ Atenção! Próximo do limite")
        else:
            st.success("✅ Dentro do recomendado")
    
    with col2:
        st.metric(
            label="Nível de Endividamento",
            value=f"{saude['endividamento']:.1f}%",
            delta="-1.2%" if saude['endividamento'] < 30 else "0%"
        )
        
        if saude['endividamento'] > 50:
            st.error("⚠️ Endividamento elevado")
        elif saude['endividamento'] > 30:
            st.warning("⚠️ Endividamento moderado")
        else:
            st.success("✅ Endividamento controlado")
    
    with col3:
        st.metric(
            label="Score de Saúde Financeira",
            value=f"{saude['score']}/100",
            delta="+5" if saude['score'] > 60 else "-2"
        )
        
        if saude['score'] >= 80:
            st.success(f"✅ {saude['classificacao']}")
        elif saude['score'] >= 60:
            st.info(f"ℹ️ {saude['classificacao']}")
        elif saude['score'] >= 40:
            st.warning(f"⚠️ {saude['classificacao']}")
        else:
            st.error(f"⚠️ {saude['classificacao']}")
    
    # Exibir gráfico de despesas
    st.markdown("### Distribuição de Despesas")
    
    grafico = gerar_grafico_despesas()
    if grafico:
        st.pyplot(grafico)
    else:
        st.info("Adicione suas despesas no diagnóstico financeiro para visualizar o gráfico.")
    
    # Exibir informações sobre dívidas
    st.markdown("### Situação das Dívidas")
    
    if st.session_state.dados_financeiros["dividas"]:
        info_dividas = calcular_tempo_quitacao_dividas()
        
        # Exibir tempo total estimado
        if info_dividas["tempo_total_meses"] == float('inf'):
            st.error("⚠️ Com as parcelas atuais, algumas dívidas nunca serão quitadas (parcelas menores que os juros).")
        else:
            meses = info_dividas["tempo_total_meses"]
            anos = meses // 12
            meses_restantes = meses % 12
            
            if anos > 0:
                st.info(f"⏱️ Tempo estimado para quitar todas as dívidas: {int(anos)} anos e {int(meses_restantes)} meses")
            else:
                st.info(f"⏱️ Tempo estimado para quitar todas as dívidas: {int(meses)} meses")
        
        # Exibir método recomendado
        metodo = sugerir_metodo_quitacao()
        st.markdown(f"**Método de quitação recomendado:** {metodo['metodo']}")
        st.markdown(f"*{metodo['explicacao']}*")
        
        # Exibir tabela de dívidas
        dados_tabela = []
        for nome, info in info_dividas["dividas"].items():
            divida = st.session_state.dados_financeiros["dividas"][nome]
            
            if info["tempo_meses"] == float('inf'):
                tempo = "Nunca (parcela < juros)"
            elif info["tempo_meses"] >= 12:
                anos = info["tempo_meses"] // 12
                meses = info["tempo_meses"] % 12
                tempo = f"{int(anos)}a {int(meses)}m"
            else:
                tempo = f"{int(info['tempo_meses'])}m"
            
            dados_tabela.append({
                "Dívida": nome,
                "Valor Total": f"R$ {divida['valor_total']:.2f}",
                "Parcela": f"R$ {divida['parcela_mensal']:.2f}",
                "Juros": f"{divida.get('taxa_juros_mensal', 0):.2f}% a.m.",
                "Tempo p/ Quitar": tempo,
                "Total de Juros": f"R$ {info['total_juros']:.2f}"
            })
        
        if dados_tabela:
            st.dataframe(pd.DataFrame(dados_tabela), hide_index=True)
    else:
        st.success("✅ Você não possui dívidas cadastradas.")
    
    # Exibir dica personalizada
    st.markdown("### Dica Personalizada")
    dica = gerar_dica_financeira_personalizada()
    st.info(f"💡 {dica}")
    
    # Exibir desafios ativos
    if st.session_state.desafios_ativos:
        st.markdown("### Desafios Ativos")
        
        for desafio in st.session_state.desafios_ativos:
            dias_restantes = (desafio["data_fim"] - datetime.now()).days
            
            st.markdown(f"""
            <div class="challenge-card">
                <h4>{desafio['titulo']}</h4>
                <p>{desafio['descricao']}</p>
                <p><strong>Dificuldade:</strong> {desafio['dificuldade']} | <strong>Pontos:</strong> {desafio['pontos']}</p>
                <p><strong>Tempo restante:</strong> {max(0, dias_restantes)} dias</p>
            </div>
            """, unsafe_allow_html=True)

def pagina_consultor():
    """
    Exibe a página do consultor virtual para tirar dúvidas e receber conselhos.
    """
    st.markdown("## 💬 Consultor Virtual Inteligente")
    
    st.markdown("""
    <div class="info-box">
        <p>Compartilhe suas preocupações financeiras e receba conselhos personalizados do nosso assistente virtual.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Abas para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Conselho Financeiro", "Simulador de Negociação", "Glossário Financeiro"])
    
    # Aba de Conselho Financeiro
    with tab1:
        st.markdown("### Conselho Financeiro Personalizado")
        st.markdown("Compartilhe sua principal preocupação financeira e receba um conselho personalizado.")
        
        preocupacao = st.text_area(
            "Qual sua principal preocupação financeira no momento?",
            placeholder="Ex: Não consigo pagar meu cartão de crédito com R$2000 em dívidas",
            height=100
        )
        
        if st.button("Obter Conselho", key="btn_conselho"):
            if preocupacao:
                with st.spinner("Gerando conselho personalizado..."):
                    conselho = gerar_conselho_financeiro(preocupacao)
                    st.markdown(f"""
                    <div class="success-box">
                        {conselho}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Por favor, descreva sua preocupação para receber um conselho.")
        
        # Exibir histórico de consultas
        if st.session_state.historico_consultas:
            st.markdown("### Histórico de Consultas")
            
            for i, consulta in enumerate(reversed(st.session_state.historico_consultas)):
                if i >= 3:  # Limitar a 3 consultas no histórico
                    break
                
                with st.expander(f"Consulta de {consulta['data'].strftime('%d/%m/%Y %H:%M')}"):
                    st.markdown(f"**Preocupação:** {consulta['preocupacao']}")
                    st.markdown(f"**Conselho:**\n{consulta['conselho']}")
    
    # Aba de Simulador de Negociação
    with tab2:
        st.markdown("### Simulador de Negociação de Dívidas")
        st.markdown(
            "Este simulador cria um diálogo de negociação para te ajudar a se preparar "
            "para uma conversa real com credores."
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            credor = st.text_input(
                "Nome do Credor",
                placeholder="Ex: Banco XYZ, Loja ABC"
            )
        
        with col2:
            valor_divida = st.number_input(
                "Valor da Dívida (R$)",
                min_value=0.0,
                step=100.0
            )
        
        with col3:
            dias_atraso = st.number_input(
                "Dias de Atraso",
                min_value=0,
                max_value=3650,
                step=30
            )
        
        if st.button("Simular Negociação", key="btn_simular"):
            if credor and valor_divida > 0:
                with st.spinner("Gerando simulação de negociação..."):
                    simulacao = simular_negociacao_divida(credor, valor_divida, dias_atraso)
                    st.markdown(f"""
                    <div class="info-box">
                        {simulacao}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Por favor, preencha o nome do credor e o valor da dívida.")
    
    # Aba de Glossário Financeiro
    with tab3:
        st.markdown("### Glossário Financeiro")
        st.markdown(
            "Tire suas dúvidas sobre termos financeiros e receba explicações simples e didáticas."
        )
        
        # Lista de termos comuns para sugestão
        termos_comuns = [
            "Juros Compostos", "CDI", "Selic", "CDB", "Tesouro Direto", 
            "Inflação", "Reserva de Emergência", "Diversificação",
            "Renda Fixa", "Renda Variável", "Ações", "FGC", "IOF", "IR",
            "Previdência Privada", "Portabilidade de Dívida", "Score de Crédito"
        ]
        
        # Campo para digitar o termo ou selecionar da lista
        termo = st.selectbox(
            "Selecione ou digite um termo financeiro",
            options=[""] + termos_comuns,
            index=0
        )
        
        # Ou digite um termo personalizado
        termo_personalizado = st.text_input(
            "Ou digite outro termo financeiro",
            placeholder="Ex: Educação Financeira, Cartão de Crédito"
        )
        
        # Usar o termo personalizado se fornecido, caso contrário usar o selecionado
        termo_final = termo_personalizado if termo_personalizado else termo
        
        if st.button("Explicar Termo", key="btn_explicar"):
            if termo_final:
                with st.spinner(f"Buscando explicação para '{termo_final}'..."):
                    explicacao = obter_explicacao_termo_financeiro(termo_final)
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>{termo_final}</h4>
                        {explicacao}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Adicionar pontos por aprender um novo termo
                    adicionar_pontos(5, f"Aprendeu sobre {termo_final}")
            else:
                st.error("Por favor, selecione ou digite um termo financeiro.")

def pagina_diagnostico():
    """
    Exibe a página de diagnóstico financeiro para coletar dados do usuário.
    """
    st.markdown("## 📝 Diagnóstico Financeiro")
    
    st.markdown("""
    <div class="info-box">
        <p>Preencha as informações abaixo para receber um diagnóstico completo da sua saúde financeira 
        e recomendações personalizadas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Criar abas para diferentes seções do diagnóstico
    tab1, tab2, tab3, tab4 = st.tabs(["Renda", "Despesas", "Dívidas", "Metas"])
    
    # Aba de Renda
    with tab1:
        st.markdown("### Informações de Renda")
        
        # Renda mensal
        renda_mensal = st.number_input(
            "Renda Mensal Total (R$)",
            min_value=0.0,
            value=st.session_state.dados_financeiros["renda_mensal"] or 0.0,
            step=100.0
        )
        
        # Reserva de emergência
        reserva_emergencia = st.number_input(
            "Reserva de Emergência (R$)",
            min_value=0.0,
            value=st.session_state.dados_financeiros.get("reserva_emergencia", 0.0),
            step=100.0,
            help="Valor total disponível em sua reserva de emergência"
        )
        
        # Outras fontes de renda
        st.markdown("#### Outras Fontes de Renda (opcional)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fonte_extra = st.text_input(
                "Descrição da Fonte de Renda",
                placeholder="Ex: Freelance, Aluguel"
            )
        
        with col2:
            valor_extra = st.number_input(
                "Valor Mensal (R$)",
                min_value=0.0,
                step=100.0
            )
        
        if st.button("Adicionar Fonte de Renda", key="btn_add_renda"):
            if fonte_extra and valor_extra > 0:
                if "outras_rendas" not in st.session_state.dados_financeiros:
                    st.session_state.dados_financeiros["outras_rendas"] = {}
                
                st.session_state.dados_financeiros["outras_rendas"][fonte_extra] = valor_extra
                st.success(f"Fonte de renda '{fonte_extra}' adicionada com sucesso!")
                
                # Limpar campos
                st.rerun()
        
        # Exibir fontes de renda cadastradas
        if "outras_rendas" in st.session_state.dados_financeiros and st.session_state.dados_financeiros["outras_rendas"]:
            st.markdown("#### Fontes de Renda Cadastradas")
            
            for fonte, valor in st.session_state.dados_financeiros["outras_rendas"].items():
                st.markdown(f"- **{fonte}**: R$ {valor:.2f}")
        
        # Botão para salvar informações de renda
        if st.button("Salvar Informações de Renda", key="btn_salvar_renda"):
            st.session_state.dados_financeiros["renda_mensal"] = renda_mensal
            st.session_state.dados_financeiros["reserva_emergencia"] = reserva_emergencia
            
            st.success("✅ Informações de renda salvas com sucesso!")
            
            # Adicionar pontos se for a primeira vez
            if not st.session_state.diagnostico_realizado:
                adicionar_pontos(20, "Iniciou seu diagnóstico financeiro")
    
    # Aba de Despesas
    with tab2:
        st.markdown("### Despesas Mensais")
        
        # Separar em despesas fixas e variáveis
        despesa_tipo = st.radio(
            "Tipo de Despesa",
            options=["Fixa", "Variável"],
            horizontal=True,
            help="Despesas fixas são aquelas que têm valor constante todo mês. Despesas variáveis podem mudar de valor ou frequência."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            categoria_despesa = st.text_input(
                "Categoria da Despesa",
                placeholder="Ex: Aluguel, Supermercado, Lazer"
            )
        
        with col2:
            valor_despesa = st.number_input(
                "Valor Mensal (R$)",
                min_value=0.0,
                step=10.0
            )
        
        if st.button("Adicionar Despesa", key="btn_add_despesa"):
            if categoria_despesa and valor_despesa > 0:
                # Determinar o dicionário correto com base no tipo
                dict_alvo = "despesas_fixas" if despesa_tipo == "Fixa" else "despesas_variaveis"
                
                # Adicionar despesa
                st.session_state.dados_financeiros[dict_alvo][categoria_despesa] = valor_despesa
                st.success(f"Despesa '{categoria_despesa}' adicionada com sucesso!")
                
                # Limpar campos
                st.rerun()
        
        # Exibir despesas cadastradas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Despesas Fixas")
            
            if st.session_state.dados_financeiros["despesas_fixas"]:
                for categoria, valor in st.session_state.dados_financeiros["despesas_fixas"].items():
                    st.markdown(f"- **{categoria}**: R$ {valor:.2f}")
                
                total_fixas = sum(st.session_state.dados_financeiros["despesas_fixas"].values())
                st.markdown(f"**Total: R$ {total_fixas:.2f}**")
            else:
                st.info("Nenhuma despesa fixa cadastrada.")
        
        with col2:
            st.markdown("#### Despesas Variáveis")
            
            if st.session_state.dados_financeiros["despesas_variaveis"]:
                for categoria, valor in st.session_state.dados_financeiros["despesas_variaveis"].items():
                    st.markdown(f"- **{categoria}**: R$ {valor:.2f}")
                
                total_variaveis = sum(st.session_state.dados_financeiros["despesas_variaveis"].values())
                st.markdown(f"**Total: R$ {total_variaveis:.2f}**")
            else:
                st.info("Nenhuma despesa variável cadastrada.")
        
        # Botão para limpar todas as despesas
        if st.button("Limpar Todas as Despesas", key="btn_limpar_despesas"):
            st.session_state.dados_financeiros["despesas_fixas"] = {}
            st.session_state.dados_financeiros["despesas_variaveis"] = {}
            st.success("✅ Todas as despesas foram removidas.")
            st.rerun()
    
    # Aba de Dívidas
    with tab3:
        st.markdown("### Dívidas Ativas")
        
        # Formulário para adicionar dívida
        col1, col2 = st.columns(2)
        
        with col1:
            nome_divida = st.text_input(
                "Nome/Descrição da Dívida",
                placeholder="Ex: Cartão Banco XYZ, Financiamento Carro"
            )
            
            valor_total = st.number_input(
                "Valor Total da Dívida (R$)",
                min_value=0.0,
                step=100.0
            )
            
            parcela_mensal = st.number_input(
                "Valor da Parcela Mensal (R$)",
                min_value=0.0,
                step=10.0
            )
        
        with col2:
            taxa_juros = st.number_input(
                "Taxa de Juros Mensal (%)",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                help="Taxa de juros mensal. Ex: 3.5 para 3,5% ao mês"
            )
            
            data_vencimento = st.date_input(
                "Data de Vencimento Mensal",
                value=None,
                help="Data de vencimento mensal da parcela"
            )
            
            total_parcelas = st.number_input(
                "Total de Parcelas",
                min_value=0,
                step=1,
                help="Número total de parcelas. Deixe 0 se não souber ou não for parcelado."
            )
        
        if st.button("Adicionar Dívida", key="btn_add_divida"):
            if nome_divida and valor_total > 0:
                # Criar dicionário com informações da dívida
                nova_divida = {
                    "valor_total": valor_total,
                    "parcela_mensal": parcela_mensal,
                    "taxa_juros_mensal": taxa_juros,
                    "data_vencimento": data_vencimento.strftime("%d/%m/%Y") if data_vencimento else None,
                    "total_parcelas": total_parcelas if total_parcelas > 0 else None
                }
                
                # Adicionar à lista de dívidas
                st.session_state.dados_financeiros["dividas"][nome_divida] = nova_divida
                st.success(f"Dívida '{nome_divida}' adicionada com sucesso!")
                
                # Limpar campos
                st.rerun()
        
        # Exibir dívidas cadastradas
        if st.session_state.dados_financeiros["dividas"]:
            st.markdown("#### Dívidas Cadastradas")
            
            for nome, info in st.session_state.dados_financeiros["dividas"].items():
                with st.expander(f"{nome} - R$ {info['valor_total']:.2f}"):
                    st.markdown(f"**Valor Total:** R$ {info['valor_total']:.2f}")
                    st.markdown(f"**Parcela Mensal:** R$ {info['parcela_mensal']:.2f}")
                    st.markdown(f"**Taxa de Juros:** {info['taxa_juros_mensal']:.2f}% ao mês")
                    
                    if info['data_vencimento']:
                        st.markdown(f"**Vencimento:** {info['data_vencimento']}")
                    
                    if info['total_parcelas']:
                        st.markdown(f"**Total de Parcelas:** {info['total_parcelas']}")
                    
                    # Botão para remover dívida
                    if st.button(f"Remover {nome}", key=f"btn_remover_{nome}"):
                        del st.session_state.dados_financeiros["dividas"][nome]
                        st.success(f"Dívida '{nome}' removida com sucesso!")
                        st.rerun()
        else:
            st.info("Nenhuma dívida cadastrada.")
    
    # Aba de Metas
    with tab4:
        st.markdown("### Metas Financeiras")
        
        # Formulário para adicionar meta
        col1, col2 = st.columns(2)
        
        with col1:
            nome_meta = st.text_input(
                "Descrição da Meta",
                placeholder="Ex: Comprar um carro, Quitar dívidas"
            )
            
            valor_meta = st.number_input(
                "Valor Necessário (R$)",
                min_value=0.0,
                step=100.0
            )
        
        with col2:
            prazo_meta = st.number_input(
                "Prazo (meses)",
                min_value=1,
                step=1
            )
            
            prioridade = st.selectbox(
                "Prioridade",
                options=["Alta", "Média", "Baixa"]
            )
        
        if st.button("Adicionar Meta", key="btn_add_meta"):
            if nome_meta and valor_meta > 0 and prazo_meta > 0:
                # Criar dicionário com informações da meta
                nova_meta = {
                    "valor": valor_meta,
                    "prazo_meses": prazo_meta,
                    "prioridade": prioridade,
                    "valor_mensal": valor_meta / prazo_meta,
                    "data_criacao": datetime.now().strftime("%d/%m/%Y")
                }
                
                # Adicionar à lista de metas
                st.session_state.dados_financeiros["metas"][nome_meta] = nova_meta
                st.success(f"Meta '{nome_meta}' adicionada com sucesso!")
                
                # Adicionar pontos e conquista
                if len(st.session_state.dados_financeiros["metas"]) == 1:
                    adicionar_pontos(15, "Definiu sua primeira meta financeira")
                    adicionar_conquista("Primeira Meta Definida! 🎯")
                
                # Limpar campos
                st.rerun()
        
        # Exibir metas cadastradas
        if st.session_state.dados_financeiros["metas"]:
            st.markdown("#### Metas Cadastradas")
            
            for nome, info in st.session_state.dados_financeiros["metas"].items():
                with st.expander(f"{nome} - R$ {info['valor']:.2f}"):
                    st.markdown(f"**Valor Total:** R$ {info['valor']:.2f}")
                    st.markdown(f"**Prazo:** {info['prazo_meses']} meses")
                    st.markdown(f"**Valor Mensal Necessário:** R$ {info['valor_mensal']:.2f}")
                    st.markdown(f"**Prioridade:** {info['prioridade']}")
                    st.markdown(f"**Data de Criação:** {info['data_criacao']}")
                    
                    # Botão para remover meta
                    if st.button(f"Remover {nome}", key=f"btn_remover_meta_{nome}"):
                        del st.session_state.dados_financeiros["metas"][nome]
                        st.success(f"Meta '{nome}' removida com sucesso!")
                        st.rerun()
        else:
            st.info("Nenhuma meta cadastrada.")
    
    # Botão para finalizar diagnóstico
    if st.button("Finalizar Diagnóstico", key="btn_finalizar"):
        # Verificar se há informações mínimas
        if not st.session_state.dados_financeiros["renda_mensal"]:
            st.error("Por favor, informe sua renda mensal na aba 'Renda'.")
        else:
            st.session_state.diagnostico_realizado = True
            
            # Adicionar pontos e conquista
            if "diagnostico_completo" not in st.session_state.conquistas:
                adicionar_pontos(30, "Completou o diagnóstico financeiro")
                adicionar_conquista("Diagnóstico Completo! 📊")
            
            st.success("✅ Diagnóstico financeiro concluído com sucesso!")
            st.balloons()
            
            # Redirecionar para o dashboard
            st.session_state.pagina_atual = "dashboard"
            st.rerun()

def pagina_desafios():
    """
    Exibe a página de desafios financeiros para o usuário.
    """
    st.markdown("## 🎯 Desafios Financeiros")
    
    st.markdown("""
    <div class="info-box">
        <p>Desafios ajudam a desenvolver hábitos financeiros saudáveis de forma divertida. 
        Complete-os para ganhar pontos e melhorar sua saúde financeira!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Botão para gerar novo desafio
    if st.button("Gerar Novo Desafio", key="btn_novo_desafio"):
        desafio = gerar_desafio_aleatorio()
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
            st.rerun()
    
    # Exibir desafios ativos
    if st.session_state.desafios_ativos:
        st.markdown("### Desafios Ativos")
        
        for i, desafio in enumerate(st.session_state.desafios_ativos):
            dias_restantes = (desafio["data_fim"] - datetime.now()).days
            
            with st.expander(f"{desafio['titulo']} ({max(0, dias_restantes)} dias restantes)"):
                st.markdown(f"**Descrição:** {desafio['descricao']}")
                st.markdown(f"**Dificuldade:** {desafio['dificuldade']}")
                st.markdown(f"**Pontos:** {desafio['pontos']}")
                st.markdown(f"**Data de Início:** {desafio['data_inicio'].strftime('%d/%m/%Y')}")
                st.markdown(f"**Data de Término:** {desafio['data_fim'].strftime('%d/%m/%Y')}")
                
                # Botão para concluir desafio
                if st.button(f"Marcar como Concluído", key=f"btn_concluir_{i}"):
                    concluir_desafio(i)
                    st.rerun()
                
                # Botão para abandonar desafio
                if st.button(f"Abandonar Desafio", key=f"btn_abandonar_{i}"):
                    st.session_state.desafios_ativos.pop(i)
                    st.warning("Desafio abandonado.")
                    st.rerun()
    else:
        st.info("Você não possui desafios ativos no momento. Gere um novo desafio para começar!")
    
    # Exibir desafios concluídos
    if st.session_state.desafios_concluidos:
        st.markdown("### Desafios Concluídos")
        
        for desafio in st.session_state.desafios_concluidos:
            st.markdown(f"""
            <div class="success-box">
                <h4>{desafio['titulo']} ✅</h4>
                <p>{desafio['descricao']}</p>
                <p><strong>Pontos ganhos:</strong> {desafio['pontos']}</p>
            </div>
            """, unsafe_allow_html=True)

def pagina_educacional():
    """
    Exibe a página de conteúdo educacional sobre finanças.
    """
    st.markdown("## 📚 Conteúdo Educacional")
    
    st.markdown("""
    <div class="info-box">
        <p>Aprenda conceitos financeiros importantes de forma simples e prática.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Módulos educacionais
    modulos = [
        {
            "titulo": "Fundamentos de Finanças Pessoais",
            "descricao": "Aprenda os conceitos básicos para organizar suas finanças.",
            "topicos": [
                "Orçamento pessoal e familiar",
                "Diferença entre necessidades e desejos",
                "Como criar uma reserva de emergência",
                "Planejamento financeiro de curto e longo prazo"
            ]
        },
        {
            "titulo": "Gestão de Dívidas",
            "descricao": "Estratégias para sair das dívidas e manter-se no azul.",
            "topicos": [
                "Como identificar dívidas prioritárias",
                "Métodos de quitação: Bola de Neve vs. Avalanche",
                "Negociação com credores",
                "Consolidação de dívidas"
            ]
        },
        {
            "titulo": "Investimentos para Iniciantes",
            "descricao": "Primeiros passos no mundo dos investimentos.",
            "topicos": [
                "Renda fixa vs. Renda variável",
                "Perfil de investidor",
                "Diversificação e risco",
                "Investimentos para diferentes objetivos"
            ]
        }
    ]
    
    # Exibir módulos em cards
    for i, modulo in enumerate(modulos):
        st.markdown(f"""
        <div class="challenge-card">
            <h3>{modulo['titulo']}</h3>
            <p>{modulo['descricao']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Expandir para mostrar tópicos
        with st.expander("Ver tópicos"):
            for topico in modulo["topicos"]:
                st.markdown(f"- {topico}")
            
            # Botão para acessar módulo
            if st.button(f"Acessar Módulo", key=f"btn_modulo_{i}"):
                st.session_state.modulo_atual = i
                
                # Adicionar pontos por acessar módulo educacional
                adicionar_pontos(5, f"Acessou o módulo {modulo['titulo']}")
                
                # Exibir conteúdo do módulo
                st.markdown(f"## {modulo['titulo']}")
                st.markdown(f"*{modulo['descricao']}*")
                
                # Conteúdo simulado do módulo
                st.markdown("### Conteúdo do Módulo")
                st.info(
                    "Este é um exemplo de conteúdo educacional. Em uma implementação completa, "
                    "aqui seriam exibidos textos, vídeos e exercícios interativos sobre o tema."
                )
                
                # Botão para marcar como concluído
                if st.button("Marcar como Concluído", key=f"btn_concluir_modulo_{i}"):
                    adicionar_pontos(15, f"Concluiu o módulo {modulo['titulo']}")
                    adicionar_conquista(f"Módulo Concluído: {modulo['titulo']}! 📚")
                    st.success(f"Módulo '{modulo['titulo']}' concluído com sucesso!")
                    st.balloons()
    
    # Dicas rápidas
    st.markdown("### Dicas Rápidas")
    
    dicas = [
        {
            "titulo": "Regra 50-30-20",
            "descricao": "Divida seu orçamento em 50% para necessidades, 30% para desejos e 20% para poupança e investimentos."
        },
        {
            "titulo": "Efeito Latte",
            "descricao": "Pequenos gastos diários (como um café) podem somar quantias significativas ao longo do tempo."
        },
        {
            "titulo": "Fundo de Emergência",
            "descricao": "Tente guardar o equivalente a 3-6 meses de despesas para emergências."
        }
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h4>{dicas[0]['titulo']}</h4>
            <p>{dicas[0]['descricao']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>{dicas[1]['titulo']}</h4>
            <p>{dicas[1]['descricao']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-box">
            <h4>{dicas[2]['titulo']}</h4>
            <p>{dicas[2]['descricao']}</p>
        </div>
        """, unsafe_allow_html=True)

def pagina_conquistas():
    """
    Exibe a página de conquistas e progresso do usuário.
    """
    st.markdown("## 🏆 Conquistas e Progresso")
    
    # Exibir informações de nível e pontos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Nível Atual: {st.session_state.nivel}")
        
        # Barra de progresso para o próximo nível
        progresso_nivel = (st.session_state.pontos % 100) / 100
        st.progress(progresso_nivel, text=f"Progresso para Nível {st.session_state.nivel + 1}: {int(progresso_nivel * 100)}%")
        
        # Pontos totais
        st.markdown(f"**Pontos Totais:** {st.session_state.pontos}")
        
        # Pontos para próximo nível
        pontos_proximo = 100 - (st.session_state.pontos % 100)
        st.markdown(f"**Pontos para o próximo nível:** {pontos_proximo}")
    
    with col2:
        # Medalha de acordo com o nível
        if st.session_state.nivel >= 5:
            st.markdown("""
            <div style="text-align: center;">
                <h1 style="font-size: 4rem;">🏆</h1>
                <h3>Especialista Financeiro</h3>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.nivel >= 3:
            st.markdown("""
            <div style="text-align: center;">
                <h1 style="font-size: 4rem;">🥈</h1>
                <h3>Estrategista Financeiro</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center;">
                <h1 style="font-size: 4rem;">🥉</h1>
                <h3>Aprendiz Financeiro</h3>
            </div>
            """, unsafe_allow_html=True)
    
    # Exibir conquistas
    st.markdown("### Suas Conquistas")
    
    if st.session_state.conquistas:
        # Organizar conquistas em colunas
        cols = st.columns(3)
        
        for i, conquista in enumerate(st.session_state.conquistas):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="success-box">
                    <h4>{conquista}</h4>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Você ainda não possui conquistas. Continue usando o aplicativo para desbloquear conquistas!")
    
    # Exibir estatísticas
    st.markdown("### Estatísticas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Consultas Realizadas",
            value=len(st.session_state.historico_consultas),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Desafios Concluídos",
            value=len(st.session_state.desafios_concluidos),
            delta=None
        )
    
    with col3:
        # Calcular dias de uso (simulado)
        dias_uso = len(st.session_state.historico_consultas) + 1
        
        st.metric(
            label="Dias de Uso",
            value=dias_uso,
            delta=None
        )
    
    # Próximas conquistas a desbloquear
    st.markdown("### Próximas Conquistas")
    
    proximas_conquistas = [
        "Complete 5 desafios financeiros 🎯",
        "Atinja o Nível 3 de experiência 🌟",
        "Mantenha despesas abaixo de 70% da renda por 30 dias 💰",
        "Crie um plano completo de quitação de dívidas 📝",
        "Aprenda 10 termos financeiros no glossário 📚"
    ]
    
    # Filtrar conquistas que o usuário ainda não possui
    for conquista in proximas_conquistas:
        if not any(c in conquista for c in st.session_state.conquistas):
            st.markdown(f"- {conquista}")

# --- Função Principal ---
def main():
    """
    Função principal que controla o fluxo da aplicação.
    """
    # Inicializar sessão
    inicializar_sessao()
    
    # Exibir cabeçalho
    exibir_cabecalho()
    
    # Exibir barra lateral
    exibir_barra_lateral()
    
    # Exibir página atual
    if st.session_state.pagina_atual == "boas_vindas":
        pagina_boas_vindas()
    elif st.session_state.pagina_atual == "dashboard":
        pagina_dashboard()
    elif st.session_state.pagina_atual == "consultor":
        pagina_consultor()
    elif st.session_state.pagina_atual == "diagnostico":
        pagina_diagnostico()
    elif st.session_state.pagina_atual == "desafios":
        pagina_desafios()
    elif st.session_state.pagina_atual == "educacional":
        pagina_educacional()
    elif st.session_state.pagina_atual == "conquistas":
        pagina_conquistas()

# Executar aplicação
if __name__ == "__main__":
    main()
