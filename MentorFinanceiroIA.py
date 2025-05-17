# Mentor Financeiro AI
# Aplicação de assistência financeira com interface Streamlit e recursos de gamificação
# Desenvolvido para fins educacionais

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
            background-color: #f0f2f6; /* Cor de fundo mais suave */
            padding: 20px;
            font-family: 'Roboto', sans-serif; /* Fonte mais moderna */
        }
        
        /* Estilo para cabeçalhos */
        h1, h2, h3 {
            color: #5c1691; /* Roxo principal para títulos */
        }
        
        /* Estilo para caixas de informação */
        .info-box {
            background-color: #e8eaf6; /* Azul claro para info */
            border-left: 5px solid #5c1691; /* Roxo */
            padding: 20px;
            border-radius: 8px; /* Bordas mais arredondadas */
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Estilo para caixas de sucesso */
        .success-box {
            background-color: #e8f5e9; /* Verde claro para sucesso */
            border-left: 5px solid #4caf50; /* Verde */
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Estilo para caixas de alerta */
        .warning-box {
            background-color: #fff3e0; /* Laranja claro para alerta */
            border-left: 5px solid #ff9800; /* Laranja */
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Estilo para barras de progresso */
        .stProgress > div > div {
            background-color: #5c1691; /* Roxo */
        }
        
        /* Estilo para botões */
        .stButton button {
            background-color: #5c1691; /* Roxo */
            color: white;
            border-radius: 20px; /* Botões mais arredondados (pílula) */
            border: none;
            padding: 10px 25px; /* Mais padding */
            font-weight: bold;
            transition: background-color 0.3s ease; /* Transição suave */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stButton button:hover {
            background-color: #4a148c; /* Roxo mais escuro no hover */
        }

        /* Estilo para botões secundários (ex: abandonar desafio) */
        .stButton button[kind="secondary"] {
            background-color: #e0e0e0; /* Cinza claro */
            color: #333333; /* Texto escuro */
        }
        .stButton button[kind="secondary"]:hover {
            background-color: #bdbdbd; /* Cinza mais escuro no hover */
        }
        
        /* Estilo para cartões de desafio */
        .challenge-card {
            background-color: #ffffff; /* Fundo branco para cards */
            border: 1px solid #e0e0e0; /* Borda sutil */
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra mais pronunciada */
        }
        
        /* Estilo para medalhas e conquistas */
        .badge {
            display: inline-block;
            background-color: #c97ffa; /* Lilás para badges */
            color: white;
            border-radius: 20px;
            padding: 5px 15px;
            margin-right: 10px;
            font-weight: bold;
            font-size: 0.9em;
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# Aplicar estilo personalizado
aplicar_estilo()

# --- Configuração da API Key do Google Generative AI ---
def configurar_api_key():
    if 'api_key_configurada' in st.session_state and st.session_state.api_key_configurada:
        return True
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    
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
    if not configurar_api_key():
        return None
    
    generation_config = {
        "temperature": 0.75,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash", # Modelo atualizado
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        st.error(f"❌ Erro ao inicializar o modelo Gemini: {e}")
        return None

# --- Inicialização da Sessão ---
def inicializar_sessao():
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

    # NOVA VARIÁVEL DE SESSÃO para o desafio proposto mas ainda não aceito
    if 'desafio_proposto' not in st.session_state:
        st.session_state.desafio_proposto = None
            
    if 'pagina_atual' not in st.session_state:
        st.session_state.pagina_atual = "boas_vindas"
    
    if 'historico_consultas' not in st.session_state:
        st.session_state.historico_consultas = []
    
    if 'diagnostico_realizado' not in st.session_state:
        st.session_state.diagnostico_realizado = False

# --- Funções de Gamificação ---
def adicionar_pontos(quantidade, motivo=""):
    st.session_state.pontos += quantidade
    nivel_anterior = st.session_state.nivel
    st.session_state.nivel = 1 + (st.session_state.pontos // 100)
    
    if st.session_state.nivel > nivel_anterior:
        adicionar_conquista(f"Nível {st.session_state.nivel} Alcançado! 🏆")
        st.balloons()
    
    if motivo:
        st.success(f"🎉 +{quantidade} pontos: {motivo}")

def adicionar_conquista(conquista):
    if conquista not in st.session_state.conquistas:
        st.session_state.conquistas.append(conquista)
        st.success(f"🏆 Nova conquista desbloqueada: {conquista}")

def gerar_desafio_aleatorio():
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
            "titulo": "Pesquisa de Preços Essenciais",
            "descricao": "Compare preços de 3 itens essenciais da sua lista de compras em 3 lugares diferentes antes de comprar.",
            "dificuldade": "Fácil",
            "pontos": 20,
            "duracao_dias": 3
        },
        {
            "titulo": "Dia Sem Gastos Supérfluos",
            "descricao": "Passe um dia inteiro sem realizar nenhum gasto que não seja absolutamente essencial (alimentação básica, transporte obrigatório).",
            "dificuldade": "Difícil",
            "pontos": 40,
            "duracao_dias": 1
        },
        {
            "titulo": "Revisão de Assinaturas",
            "descricao": "Revise todas as suas assinaturas mensais (streaming, apps, etc.) e cancele pelo menos uma que não usa com frequência.",
            "dificuldade": "Médio",
            "pontos": 35,
            "duracao_dias": 2
        }
    ]
    
    desafio = random.choice(desafios)
    data_inicio = datetime.now()
    data_fim = data_inicio + timedelta(days=desafio["duracao_dias"])
    
    desafio["data_inicio"] = data_inicio
    desafio["data_fim"] = data_fim
    desafio["concluido"] = False
    
    return desafio

def aceitar_desafio(desafio):
    titulos_ativos = [d["titulo"] for d in st.session_state.desafios_ativos]
    if desafio["titulo"] not in titulos_ativos:
        st.session_state.desafios_ativos.append(desafio)
        st.success(f"🎯 Desafio aceito: {desafio['titulo']}")
        adicionar_pontos(5, "Aceitou um novo desafio")

def concluir_desafio(indice):
    if 0 <= indice < len(st.session_state.desafios_ativos):
        desafio = st.session_state.desafios_ativos[indice]
        desafio["concluido"] = True
        
        st.session_state.desafios_concluidos.append(desafio)
        st.session_state.desafios_ativos.pop(indice)
        
        adicionar_pontos(desafio["pontos"], f"Concluiu o desafio: {desafio['titulo']}")
        
        if len(st.session_state.desafios_concluidos) == 1:
            adicionar_conquista("Primeiro Desafio Concluído! 🌟")
        elif len(st.session_state.desafios_concluidos) == 5:
            adicionar_conquista("Desafiador Experiente: 5 Desafios Concluídos! 🔥")
        elif len(st.session_state.desafios_concluidos) == 10:
            adicionar_conquista("Mestre dos Desafios: 10 Desafios Concluídos! 🏅")

# --- Funções de Diagnóstico Financeiro (sem alterações) ---
def calcular_saude_financeira():
    dados = st.session_state.dados_financeiros
    resultado = {
        "comprometimento_renda": 0, "endividamento": 0, "reserva_emergencia": 0,
        "score": 0, "classificacao": "Não disponível"
    }
    if not dados["renda_mensal"]: return resultado
    total_despesas_fixas = sum(dados["despesas_fixas"].values()) if dados["despesas_fixas"] else 0
    total_despesas_variaveis = sum(dados["despesas_variaveis"].values()) if dados["despesas_variaveis"] else 0
    total_parcelas_dividas = sum(d.get("parcela_mensal", 0) for d in dados["dividas"].values() if d.get("parcela_mensal"))
    total_despesas = total_despesas_fixas + total_despesas_variaveis + total_parcelas_dividas
    if dados["renda_mensal"] > 0:
        resultado["comprometimento_renda"] = (total_despesas / dados["renda_mensal"]) * 100
    total_dividas = sum(d.get("valor_total", 0) for d in dados["dividas"].values())
    if dados["renda_mensal"] > 0:
        resultado["endividamento"] = (total_dividas / (dados["renda_mensal"] * 12)) * 100 if dados["renda_mensal"] * 12 > 0 else float('inf')
    reserva = dados.get("reserva_emergencia", 0)
    if total_despesas > 0:
        resultado["reserva_emergencia"] = reserva / total_despesas if reserva else 0
    score = 100
    if resultado["comprometimento_renda"] > 80: score -= 40
    elif resultado["comprometimento_renda"] > 60: score -= 25
    elif resultado["comprometimento_renda"] > 40: score -= 10
    if resultado["endividamento"] > 50: score -= 30
    elif resultado["endividamento"] > 30: score -= 20
    elif resultado["endividamento"] > 15: score -= 10
    if resultado["reserva_emergencia"] >= 6: score += 20
    elif resultado["reserva_emergencia"] >= 3: score += 10
    elif resultado["reserva_emergencia"] < 1: score -= 20
    resultado["score"] = max(0, min(100, score))
    if resultado["score"] >= 80: resultado["classificacao"] = "Excelente"
    elif resultado["score"] >= 60: resultado["classificacao"] = "Boa"
    elif resultado["score"] >= 40: resultado["classificacao"] = "Regular"
    elif resultado["score"] >= 20: resultado["classificacao"] = "Preocupante"
    else: resultado["classificacao"] = "Crítica"
    return resultado

def gerar_grafico_despesas():
    dados = st.session_state.dados_financeiros
    todas_despesas = {}
    for categoria, valor in dados["despesas_fixas"].items(): todas_despesas[f"Fixo: {categoria}"] = valor
    for categoria, valor in dados["despesas_variaveis"].items(): todas_despesas[f"Variável: {categoria}"] = valor
    for nome_divida, info_divida in dados["dividas"].items():
        if "parcela_mensal" in info_divida and info_divida["parcela_mensal"]:
            todas_despesas[f"Dívida: {nome_divida}"] = info_divida["parcela_mensal"]
    if not todas_despesas: return None
    fig, ax = plt.subplots(figsize=(10, 6))
    despesas_ordenadas = dict(sorted(todas_despesas.items(), key=lambda x: x[1], reverse=True))
    cores = plt.cm.viridis(np.linspace(0, 1, len(despesas_ordenadas))) # Paleta de cores diferente
    wedges, texts, autotexts = ax.pie(
        despesas_ordenadas.values(), labels=None, autopct='%1.1f%%',
        startangle=90, colors=cores, wedgeprops=dict(width=0.4, edgecolor='w')) # Donut chart
    for autotext in autotexts:
        autotext.set_color('black'); autotext.set_fontsize(9); autotext.set_fontweight('bold')
    ax.legend(wedges, despesas_ordenadas.keys(), title="Categorias", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize='small')
    ax.set_title("Distribuição de Despesas Mensais", fontsize=16, pad=20, color="#5c1691")
    plt.tight_layout()
    return fig

def calcular_tempo_quitacao_dividas():
    dados = st.session_state.dados_financeiros
    resultado = {"dividas": {}, "tempo_total_meses": 0, "valor_total": 0, "juros_total": 0}
    if not dados["dividas"]: return resultado
    tempo_maximo = 0
    for nome, divida in dados["dividas"].items():
        if "valor_total" in divida and "parcela_mensal" in divida and divida["parcela_mensal"] > 0:
            valor_total = divida["valor_total"]; parcela_mensal = divida["parcela_mensal"]
            taxa_juros_mensal = divida.get("taxa_juros_mensal", 0) / 100 if "taxa_juros_mensal" in divida else 0
            saldo_devedor = valor_total; meses = 0; total_pago = 0; total_juros = 0
            while saldo_devedor > 0.01 and meses < 600: # Limite e condição de parada
                meses += 1; juros_mes = saldo_devedor * taxa_juros_mensal; total_juros += juros_mes
                amortizacao = parcela_mensal - juros_mes
                if amortizacao <= 0: meses = float('inf'); break # Parcela não cobre juros
                pagamento_efetivo = min(parcela_mensal, saldo_devedor + juros_mes)
                total_pago += pagamento_efetivo
                saldo_devedor += juros_mes - pagamento_efetivo
            resultado["dividas"][nome] = {"tempo_meses": meses, "total_pago": total_pago, "total_juros": total_juros}
            resultado["valor_total"] += valor_total; resultado["juros_total"] += total_juros
            if meses != float('inf'): tempo_maximo = max(tempo_maximo, meses)
            else: tempo_maximo = float('inf') # Se uma dívida é infinita, o total é infinito
    resultado["tempo_total_meses"] = tempo_maximo
    return resultado

def sugerir_metodo_quitacao():
    dados = st.session_state.dados_financeiros
    if not dados["dividas"]: return {"metodo": "Nenhum", "explicacao": "Não há dívidas cadastradas."}
    total_dividas_valor = sum(d.get("valor_total", 0) for d in dados["dividas"].values())
    dividas_com_juros_altos = sum(1 for d in dados["dividas"].values() if d.get("taxa_juros_mensal", 0) > 3) # Juros > 3% a.m.
    quantidade_dividas = len(dados["dividas"])
    perfil_motivacional = "conquistas_rapidas"
    if dividas_com_juros_altos > 0 and (dividas_com_juros_altos >= quantidade_dividas / 2 or total_dividas_valor > 5000):
        metodo = "Avalanche (Foco nos Juros Altos)"
        explicacao = "Priorize a dívida com a MAIOR taxa de juros. Isso economiza mais dinheiro a longo prazo, especialmente com juros altos envolvidos."
    elif perfil_motivacional == "conquistas_rapidas" and quantidade_dividas > 1:
        metodo = "Bola de Neve (Foco na Menor Dívida)"
        explicacao = "Priorize a dívida com o MENOR saldo devedor. Quitar dívidas rapidamente pode aumentar sua motivação para continuar."
    else:
        metodo = "Personalizado/Híbrido"
        explicacao = "Analise suas dívidas. Se tiver uma pequena fácil de quitar, comece por ela para ganhar ânimo (Bola de Neve). Depois, ataque as com juros mais altos (Avalanche)."
    return {"metodo": metodo, "explicacao": explicacao}

# --- Funções de Conteúdo Educacional (sem alterações) ---
def obter_explicacao_termo_financeiro(termo):
    modelo = configurar_modelo_gemini()
    if not modelo: return "Não foi possível obter a explicação. Verifique a API Key."
    try:
        prompt = [f"Explique o termo financeiro '{termo}' de forma simples e didática para um leigo em finanças. Use no máximo 2 parágrafos e um exemplo prático. Responda em português do Brasil."]
        response = modelo.generate_content(prompt)
        return response.text
    except Exception as e: return f"Erro ao obter explicação: {e}"

def gerar_dica_financeira_personalizada():
    modelo = configurar_modelo_gemini()
    if not modelo: return "Não foi possível gerar uma dica. Verifique a API Key."
    dados = st.session_state.dados_financeiros; saude = calcular_saude_financeira()
    try:
        prompt_parts = [
            f"Gere uma dica financeira personalizada e acionável (máximo 2 frases) para alguém com:",
            f"- Comprometimento de renda: {saude['comprometimento_renda']:.1f}%",
            f"- Nível de endividamento: {saude['endividamento']:.1f}% ({saude['classificacao']})",
        ]
        if dados["dividas"]: prompt_parts.append(f"- Dívidas: {', '.join(dados['dividas'].keys())}")
        else: prompt_parts.append("- Sem dívidas ativas.")
        if saude["reserva_emergencia"] > 0: prompt_parts.append(f"- Reserva para {saude['reserva_emergencia']:.1f} meses.")
        else: prompt_parts.append("- Sem reserva de emergência.")
        prompt_parts.append("A dica deve ser motivadora. Responda em português do Brasil.")
        response = modelo.generate_content(prompt_parts)
        return response.text
    except Exception as e: return f"Erro ao gerar dica: {e}"

def gerar_planejamento_financeiro(preocupacao):
    modelo = configurar_modelo_gemini()
    if not modelo: return "Não foi possível gerar um planejamento. Verifique a API Key."
    dados = st.session_state.dados_financeiros; nome = st.session_state.nome_usuario
    try:
        prompt_parts = [
            f"Sou {nome}. Minha principal preocupação financeira é: '{preocupacao}'.",
            f"Minha renda mensal: R${dados['renda_mensal']:.2f}." if dados['renda_mensal'] else "Renda mensal não informada.",
        ]
        if dados['dividas']:
            prompt_parts.append("Minhas dívidas:")
            for nome_divida, info in dados['dividas'].items():
                prompt_parts.append(f"- {nome_divida}: R${info.get('valor_total',0):.2f}, parcela R${info.get('parcela_mensal',0):.2f}, juros {info.get('taxa_juros_mensal',0):.1f}% a.m.")
        prompt_parts.extend([
            "\nVocê é um consultor financeiro experiente, empático e motivador.",
            "Preciso de um plano de ação detalhado e prático para lidar com essa situação, em português do Brasil:",
            "1. Mensagem curta de encorajamento (1-2 frases).",
            "2. Análise breve da situação com base nos dados fornecidos.",
            "3. Plano de Ação Passo-a-Passo (numerado), com sugestões concretas, incluindo valores se possível (ex: economizar X, direcionar Y para dívida Z).",
            "4. Se a renda for insuficiente, sugira 1-2 ideias realistas de renda extra adequadas ao contexto brasileiro.",
            "5. Dica final motivadora (1 frase).",
            "Seja claro, direto, use linguagem acessível. Formate com Markdown (negrito, listas)."
        ])
        response = modelo.generate_content(prompt_parts)
        st.session_state.historico_consultas.append({"data": datetime.now(), "preocupacao": preocupacao, "planejamento": response.text})
        adicionar_pontos(10, "Solicitou um planejamento financeiro")
        return response.text
    except Exception as e: return f"Erro ao gerar planejamento: {e}"

def simular_negociacao_divida(credor, valor_divida, dias_atraso):
    modelo = configurar_modelo_gemini()
    if not modelo: return "Não foi possível simular. Verifique a API Key."
    nome = st.session_state.nome_usuario
    try:
        prompt_parts = [
            f"Simule um diálogo de negociação de dívida entre {nome} (cliente) e um atendente do(a) {credor}.",
            f"Valor original da dívida: R${valor_divida:.2f}, Atraso: {dias_atraso} dias.",
            "\nO diálogo deve ser realista e incluir:",
            "1. Saudação do atendente e verificação de dados.",
            f"2. {nome} explicando a situação e o desejo de negociar.",
            "3. Atendente apresentando opções (com juros/multas, se aplicável).",
            f"4. {nome} argumentando por melhores condições (desconto, parcelamento sem juros abusivos).",
            "5. Atendente fazendo uma contraproposta.",
            "6. Fechamento do acordo ou próximos passos.",
            "Inclua dicas entre parênteses para {nome} (ex: (Mantenha a calma), (Peça o CET)).",
            "Formate como um diálogo. Responda em português do Brasil."
        ]
        response = modelo.generate_content(prompt_parts)
        adicionar_pontos(15, "Realizou uma simulação de negociação")
        return response.text
    except Exception as e: return f"Erro ao simular negociação: {e}"

# --- Componentes da Interface (sem grandes alterações, exceto talvez chaves de botões se necessário) ---
def exibir_cabecalho():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1 style='color: #5c1691; font-weight: 700;'>🌟 Mentor Financeiro AI</h1>", unsafe_allow_html=True)
        st.markdown("#### Seu assistente para trilhar o caminho da saúde financeira!")
    with col2:
        if st.session_state.nome_usuario:
            st.markdown(f"<p style='text-align: right; margin-bottom: 0px;'>Olá, <strong>{st.session_state.nome_usuario}</strong>!</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: right; margin-bottom: 15px;'>Nível: {st.session_state.nivel} | Pontos: {st.session_state.pontos}</p>", unsafe_allow_html=True)
            medalha_html = ""
            if st.session_state.nivel >= 5: medalha_html = "<div class='badge' style='background-color: #ffd700; color: #333; float: right;'>🏆 Especialista</div>"
            elif st.session_state.nivel >= 3: medalha_html = "<div class='badge' style='background-color: #c0c0c0; float: right;'>🥈 Estrategista</div>"
            elif st.session_state.nivel >= 1: medalha_html = "<div class='badge' style='background-color: #cd7f32; float: right;'>🥉 Aprendiz</div>"
            if medalha_html: st.markdown(medalha_html, unsafe_allow_html=True)

def exibir_barra_lateral():
    st.sidebar.title("Menu")
    if st.session_state.nome_usuario:
        botoes_menu = {
            "dashboard": "📊 Dashboard", "consultor": "💬 Consultor Virtual",
            "diagnostico": "📝 Diagnóstico", "desafios": "🎯 Desafios",
            "educacional": "📚 Conteúdo", "conquistas": "🏆 Conquistas"
        }
        for pagina_id, nome_botao in botoes_menu.items():
            if st.sidebar.button(nome_botao, use_container_width=True, key=f"btn_nav_{pagina_id}"):
                st.session_state.pagina_atual = pagina_id
                st.session_state.desafio_proposto = None # Limpa desafio proposto ao navegar
                st.rerun() # Garante que a página correta seja exibida
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Seu Progresso")
        progresso_nivel = (st.session_state.pontos % 100) / 100
        st.sidebar.progress(progresso_nivel, text=f"Nível {st.session_state.nivel}: {st.session_state.pontos % 100}/100 pts")
        if st.session_state.conquistas:
            st.sidebar.markdown("#### Conquistas Recentes")
            for conquista in reversed(st.session_state.conquistas[-2:]): # Mostrar as 2 últimas
                st.sidebar.markdown(f"<span class='badge'>{conquista}</span>", unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configurações")
    if 'api_key_configurada' in st.session_state and st.session_state.api_key_configurada:
        st.sidebar.success("✅ API Gemini configurada")
    else:
        st.sidebar.warning("⚠️ API Gemini não configurada")
        configurar_api_key() # Tenta configurar se não estiver
    
    st.sidebar.markdown("---")
    st.sidebar.info("Mentor Financeiro AI\n\nDesenvolvido para fins educacionais. Conteúdo de IA pode conter erros. Procure ajuda profissional.")

# --- Páginas da Aplicação ---
def pagina_boas_vindas():
    st.markdown("## 👋 Bem-vindo ao Mentor Financeiro AI!")
    st.markdown("""
    <div class="info-box">
        <h3>O que é o Mentor Financeiro AI?</h3>
        <p>Sou um assistente inteligente para ajudar você a melhorar sua saúde financeira com planejamentos, diagnósticos e desafios gamificados.</p>
        <p>Minha missão é auxiliar no primeiro passo rumo a uma vida financeira mais leve e tranquila, especialmente para quem enfrenta endividamento.</p>        
    </div>
    """, unsafe_allow_html=True)
    nome = st.text_input("Para começarmos, qual é o seu nome?", key="input_nome_boas_vindas", value=st.session_state.get("nome_usuario", ""))
    if st.button("🚀 Começar Jornada", key="btn_comecar_jornada"):
        if nome:
            st.session_state.nome_usuario = nome
            st.session_state.pagina_atual = "dashboard"
            if "Início da Jornada Financeira! 🚀" not in st.session_state.conquistas:
                 adicionar_conquista("Início da Jornada Financeira! 🚀")
                 adicionar_pontos(10, "Iniciou sua jornada financeira")
            st.success(f"Olá, {nome}! Bem-vindo à sua jornada financeira!")
            st.balloons()
            time.sleep(1) # Pequena pausa para o usuário ver a mensagem
            st.rerun()
        else:
            st.error("Por favor, informe seu nome para continuar.")
    st.markdown("### Recursos disponíveis:")
    col1, col2 = st.columns(2)
    with col1: st.markdown("- **Consultor Virtual Inteligente**\n- **Diagnóstico Financeiro Completo**\n- **Planos de Ação Personalizados**")
    with col2: st.markdown("- **Desafios Financeiros Gamificados**\n- **Conteúdo Educacional Prático**\n- **Sistema de Pontos e Conquistas**")

def pagina_dashboard():
    st.markdown("## 📊 Dashboard")
    if not st.session_state.dados_financeiros.get("renda_mensal"): # Usar .get para evitar KeyError
        st.warning("Você ainda não completou seu diagnóstico financeiro. Complete-o para visualizar seu dashboard completo.")
        if st.button("Ir para Diagnóstico Financeiro", key="btn_goto_diag_dash"):
            st.session_state.pagina_atual = "diagnostico"
            st.rerun()
        st.markdown("### Dica do Dia")
        st.info("💡 **Comece anotando todos os seus gastos!** O primeiro passo para melhorar sua saúde financeira é entender para onde seu dinheiro está indo. Anote tudo por uma semana.")
        return

    saude = calcular_saude_financeira()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Comprometimento de Renda", value=f"{saude['comprometimento_renda']:.1f}%", 
                  help="Ideal abaixo de 60-70%. Percentual da sua renda usado para pagar todas as despesas e dívidas.")
        if saude['comprometimento_renda'] > 80: st.error("⚠️ Acima do recomendado (80%)")
        elif saude['comprometimento_renda'] > 60: st.warning("⚠️ Atenção! Próximo do limite")
        else: st.success("✅ Dentro do recomendado")
    with col2:
        st.metric(label="Nível de Endividamento Total", value=f"{saude['endividamento']:.1f}%",
                  help="Ideal abaixo de 30-40% do seu patrimônio ou renda anual. Relação entre o total de suas dívidas e sua renda anual.")
        if saude['endividamento'] > 50: st.error("⚠️ Endividamento elevado")
        elif saude['endividamento'] > 30: st.warning("⚠️ Endividamento moderado")
        else: st.success("✅ Endividamento controlado")
    with col3:
        st.metric(label="Score de Saúde Financeira", value=f"{saude['score']}/100", delta=f"{saude['classificacao']}")
        if saude['score'] >= 80: st.success(f"Classificação: {saude['classificacao']}")
        elif saude['score'] >= 40: st.warning(f"Classificação: {saude['classificacao']}")
        else: st.error(f"Classificação: {saude['classificacao']}")

    st.markdown("### Distribuição de Despesas")
    grafico = gerar_grafico_despesas()
    if grafico: st.pyplot(grafico)
    else: st.info("Adicione suas despesas no diagnóstico para visualizar o gráfico.")

    st.markdown("### Situação das Dívidas")
    if st.session_state.dados_financeiros["dividas"]:
        info_dividas = calcular_tempo_quitacao_dividas()
        if info_dividas["tempo_total_meses"] == float('inf'):
            st.error("⚠️ Com as parcelas atuais, algumas dívidas podem levar muito tempo ou nunca serem quitadas (parcelas menores que os juros). Revise os valores!")
        elif info_dividas["tempo_total_meses"] > 0:
            meses, anos = info_dividas["tempo_total_meses"], info_dividas["tempo_total_meses"] // 12
            st.info(f"⏱️ Tempo estimado para quitar todas as dívidas: {int(anos)} anos e {int(meses % 12)} meses.")
        else:
             st.success("✅ Parece que não há dívidas com tempo de quitação calculado ou todas já foram quitadas!")

        metodo = sugerir_metodo_quitacao()
        st.markdown(f"**Método de quitação recomendado:** {metodo['metodo']}")
        st.markdown(f"*{metodo['explicacao']}*")
        
        dados_tabela_div = []
        for nome, info in info_dividas["dividas"].items():
            div_orig = st.session_state.dados_financeiros["dividas"][nome]
            tempo_str = f"{int(info['tempo_meses'] // 12)}a {int(info['tempo_meses'] % 12)}m" if info['tempo_meses'] != float('inf') and info['tempo_meses'] > 0 else ("Nunca" if info['tempo_meses'] == float('inf') else "N/A")
            dados_tabela_div.append({
                "Dívida": nome, "Valor Total": f"R$ {div_orig.get('valor_total',0):.2f}", 
                "Parcela": f"R$ {div_orig.get('parcela_mensal',0):.2f}", "Juros (% a.m.)": f"{div_orig.get('taxa_juros_mensal',0):.2f}",
                "Tempo p/ Quitar": tempo_str, "Total de Juros Pago": f"R$ {info.get('total_juros',0):.2f}"
            })
        if dados_tabela_div: st.dataframe(pd.DataFrame(dados_tabela_div), hide_index=True, use_container_width=True)
    else:
        st.success("✅ Você não possui dívidas cadastradas.")

    st.markdown("### Dica Personalizada do Mentor AI")
    with st.spinner("Gerando sua dica personalizada..."):
        dica = gerar_dica_financeira_personalizada()
    st.markdown(f"<div class='info-box' style='background-color: #fff9c4; border-left-color: #fbc02d;'>💡 <strong>{dica}</strong></div>", unsafe_allow_html=True)

    if st.session_state.desafios_ativos:
        st.markdown("### Seus Desafios Ativos")
        for desafio in st.session_state.desafios_ativos:
            dias_restantes = (desafio["data_fim"] - datetime.now()).days
            st.markdown(f"""
            <div class="challenge-card" style='border-left: 5px solid #5c1691;'>
                <h4>{desafio['titulo']}</h4> <p>{desafio['descricao']}</p>
                <p><strong>Dificuldade:</strong> {desafio['dificuldade']} | <strong>Pontos:</strong> {desafio['pontos']} | <strong>Restam:</strong> {max(0, dias_restantes)} dias</p>
            </div>""", unsafe_allow_html=True)

def pagina_consultor():
    st.markdown("## 💬 Consultor Virtual Inteligente")
    st.markdown("<div class='info-box'><p>Use a inteligência artificial para obter planejamentos, simular negociações e entender termos financeiros.</p></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Planejamento Financeiro", "🗣️ Simulador de Negociação", "📖 Glossário Financeiro"])
    
    with tab1:
        st.markdown("### Planejamento Financeiro Personalizado")
        preocupacao = st.text_area("Qual sua principal preocupação ou objetivo financeiro no momento?", 
                                   placeholder="Ex: Quitar o cartão de crédito de R$2000; Economizar para uma viagem; Organizar minhas finanças.", height=100, key="text_area_preocupacao")
        if st.button("💡 Obter Planejamento", key="btn_obter_planejamento"):
            if preocupacao:
                if not st.session_state.dados_financeiros.get("renda_mensal"):
                    st.warning("Para um planejamento mais preciso, preencha seus dados na seção 'Diagnóstico Financeiro' primeiro.")
                with st.spinner("Gerando seu planejamento personalizado... Isso pode levar um momento."):
                    planejamento = gerar_planejamento_financeiro(preocupacao)
                st.markdown(f"<div class='success-box'>{planejamento}</div>", unsafe_allow_html=True)
            else: st.error("Por favor, descreva sua preocupação ou objetivo.")
        if st.session_state.historico_consultas:
            st.markdown("--- \n### Histórico de Planejamentos")
            for i, consulta in enumerate(reversed(st.session_state.historico_consultas[-3:])): # Mostrar os 3 últimos
                with st.expander(f"Planejamento de {consulta['data'].strftime('%d/%m/%Y %H:%M')} - Foco: {consulta['preocupacao'][:30]}..."):
                    st.markdown(consulta['planejamento'])
    with tab2:
        st.markdown("### Simulador de Negociação de Dívidas")
        st.markdown("Prepare-se para conversas reais com credores simulando uma negociação aqui.")
        col1, col2, col3 = st.columns([2,1,1])
        with col1: credor = st.text_input("Nome do Credor", placeholder="Ex: Banco XYZ, Loja ABC", key="input_credor_sim")
        with col2: valor_divida = st.number_input("Valor da Dívida (R$)", min_value=0.0, step=100.0, key="num_valor_div_sim")
        with col3: dias_atraso = st.number_input("Dias de Atraso", min_value=0, step=1, key="num_dias_atraso_sim")
        if st.button("🤝 Simular Negociação", key="btn_simular_neg"):
            if credor and valor_divida > 0:
                with st.spinner("Gerando simulação de negociação..."):
                    simulacao = simular_negociacao_divida(credor, valor_divida, dias_atraso)
                st.markdown(f"<div class='info-box'>{simulacao}</div>", unsafe_allow_html=True)
            else: st.error("Preencha o nome do credor e o valor da dívida.")
    with tab3:
        st.markdown("### Glossário Financeiro")
        st.markdown("Entenda termos do mundo das finanças de forma clara.")
        termos_comuns = ["Juros Compostos", "CDI", "Selic", "CDB", "Tesouro Direto", "Inflação", "Reserva de Emergência", "Diversificação", "Renda Fixa", "Renda Variável", "Ações", "FGC", "IOF", "IR", "Previdência Privada", "Portabilidade de Dívida", "Score de Crédito", "CET (Custo Efetivo Total)"]
        termo_selecionado = st.selectbox("Selecione um termo comum:", options=[""] + sorted(termos_comuns), index=0, key="select_termo_glossario")
        termo_digitado = st.text_input("Ou digite um termo para buscar:", placeholder="Ex: Amortização", key="input_termo_glossario")
        termo_final = termo_digitado if termo_digitado else termo_selecionado
        if st.button("🔍 Explicar Termo", key="btn_explicar_termo"):
            if termo_final:
                with st.spinner(f"Buscando explicação para '{termo_final}'..."):
                    explicacao = obter_explicacao_termo_financeiro(termo_final)
                st.markdown(f"<div class='info-box'><h4>{termo_final}</h4>{explicacao}</div>", unsafe_allow_html=True)
                adicionar_pontos(5, f"Aprendeu sobre {termo_final}")
            else: st.error("Selecione ou digite um termo.")

def pagina_diagnostico():
    st.markdown("## 📝 Diagnóstico Financeiro")
    st.markdown("<div class='info-box'><p>Preencha suas informações para um diagnóstico completo e recomendações personalizadas. Quanto mais detalhes, melhor a análise!</p></div>", unsafe_allow_html=True)
    
    tab_renda, tab_despesas, tab_dividas, tab_metas = st.tabs(["💰 Renda", "💸 Despesas", "📉 Dívidas", "🎯 Metas"])

    # Aba de Renda
    with tab_renda:
        st.markdown("### Suas Fontes de Renda")
        renda_mensal = st.number_input("Renda Mensal Principal (Líquida) (R$)", min_value=0.0, 
                                       value=st.session_state.dados_financeiros.get("renda_mensal", 0.0), step=100.0, key="num_renda_principal")
        reserva_emergencia = st.number_input("Valor Atual da Reserva de Emergência (R$)", min_value=0.0, 
                                             value=st.session_state.dados_financeiros.get("reserva_emergencia", 0.0), step=100.0, key="num_reserva_emerg")
        if st.button("Salvar Renda e Reserva", key="btn_salvar_renda_diag"):
            st.session_state.dados_financeiros["renda_mensal"] = renda_mensal
            st.session_state.dados_financeiros["reserva_emergencia"] = reserva_emergencia
            st.success("✅ Renda e reserva salvas!")
            if not st.session_state.diagnostico_realizado and renda_mensal > 0:
                adicionar_pontos(10, "Informou sua renda no diagnóstico")

    # Aba de Despesas
    with tab_despesas:
        st.markdown("### Suas Despesas Mensais")
        tipo_despesa = st.radio("Tipo de Despesa:", ["Fixa Essencial", "Fixa Não Essencial", "Variável Essencial", "Variável Não Essencial"], horizontal=True, key="radio_tipo_despesa")
        col_desc_desp, col_val_desp = st.columns(2)
        with col_desc_desp: categoria_despesa = st.text_input("Descrição da Despesa", placeholder="Ex: Aluguel, Supermercado, Lazer", key="input_cat_despesa")
        with col_val_desp: valor_despesa = st.number_input("Valor Mensal (R$)", min_value=0.0, step=10.0, key="num_valor_despesa")
        
        if st.button("Adicionar Despesa", key="btn_add_despesa_diag"):
            if categoria_despesa and valor_despesa > 0:
                dict_alvo = "despesas_fixas" if "Fixa" in tipo_despesa else "despesas_variaveis"
                # Poderia adicionar subcategorias para essencial/não essencial se quisesse mais granularidade
                st.session_state.dados_financeiros[dict_alvo][f"{categoria_despesa} ({tipo_despesa.split(' ')[0]})"] = valor_despesa
                st.success(f"Despesa '{categoria_despesa}' adicionada!")
                # st.rerun() # Para limpar campos, mas pode ser chato para o usuário
            else: st.error("Preencha a descrição e o valor da despesa.")

        col_fixas, col_variaveis = st.columns(2)
        with col_fixas:
            st.markdown("#### Despesas Fixas Cadastradas")
            if st.session_state.dados_financeiros["despesas_fixas"]:
                for cat, val in st.session_state.dados_financeiros["despesas_fixas"].items(): st.write(f"- {cat}: R$ {val:.2f}")
                st.markdown(f"**Total Fixas: R$ {sum(st.session_state.dados_financeiros['despesas_fixas'].values()):.2f}**")
            else: st.caption("Nenhuma despesa fixa.")
        with col_variaveis:
            st.markdown("#### Despesas Variáveis Cadastradas")
            if st.session_state.dados_financeiros["despesas_variaveis"]:
                for cat, val in st.session_state.dados_financeiros["despesas_variaveis"].items(): st.write(f"- {cat}: R$ {val:.2f}")
                st.markdown(f"**Total Variáveis: R$ {sum(st.session_state.dados_financeiros['despesas_variaveis'].values()):.2f}**")
            else: st.caption("Nenhuma despesa variável.")
        if st.button("🗑️ Limpar Todas as Despesas", key="btn_limpar_td_despesas", type="secondary"):
            st.session_state.dados_financeiros["despesas_fixas"] = {}
            st.session_state.dados_financeiros["despesas_variaveis"] = {}
            st.success("Todas as despesas foram removidas.")
            # st.rerun()

    # Aba de Dívidas
    with tab_dividas:
        st.markdown("### Suas Dívidas Ativas")
        with st.form(key="form_add_divida"):
            nome_divida = st.text_input("Nome/Descrição da Dívida", placeholder="Ex: Cartão Banco XPTO, Financiamento Veículo")
            col_val_tot, col_parc = st.columns(2)
            with col_val_tot: valor_total_div = st.number_input("Valor Total da Dívida (R$)", min_value=0.0, step=100.0)
            with col_parc: parcela_mensal_div = st.number_input("Valor da Parcela Mensal (R$)", min_value=0.0, step=10.0)
            col_juros, col_prazo = st.columns(2)
            with col_juros: taxa_juros_div = st.number_input("Taxa de Juros Mensal (%)", min_value=0.0, max_value=25.0, step=0.1, format="%.2f")
            with col_prazo: total_parcelas_div = st.number_input("Total de Parcelas Restantes", min_value=0, step=1)
            submit_divida = st.form_submit_button("Adicionar Dívida")

            if submit_divida:
                if nome_divida and valor_total_div > 0:
                    st.session_state.dados_financeiros["dividas"][nome_divida] = {
                        "valor_total": valor_total_div, "parcela_mensal": parcela_mensal_div,
                        "taxa_juros_mensal": taxa_juros_div, 
                        "total_parcelas": total_parcelas_div if total_parcelas_div > 0 else None
                    }
                    st.success(f"Dívida '{nome_divida}' adicionada!")
                    # st.rerun() # Para limpar form, mas pode ser chato
                else: st.error("Preencha nome e valor total da dívida.")
        
        if st.session_state.dados_financeiros["dividas"]:
            st.markdown("#### Dívidas Cadastradas")
            for nome, info in st.session_state.dados_financeiros["dividas"].items():
                exp = st.expander(f"{nome} - Saldo: R$ {info.get('valor_total',0):.2f} / Parcela: R$ {info.get('parcela_mensal',0):.2f}")
                exp.write(f"Juros: {info.get('taxa_juros_mensal',0):.2f}% a.m. | Parcelas Restantes: {info.get('total_parcelas','N/A')}")
                if exp.button(f"Remover {nome}", key=f"rem_div_{nome.replace(' ','_')}", type="secondary"):
                    del st.session_state.dados_financeiros["dividas"][nome]
                    st.success(f"Dívida '{nome}' removida.")
                    st.rerun()
        else: st.caption("Nenhuma dívida cadastrada.")

    # Aba de Metas
    with tab_metas:
        st.markdown("### Suas Metas Financeiras")
        with st.form(key="form_add_meta"):
            nome_meta = st.text_input("Descrição da Meta", placeholder="Ex: Viagem para a praia, Comprar um notebook")
            col_val_meta, col_prazo_meta = st.columns(2)
            with col_val_meta: valor_meta = st.number_input("Valor Necessário (R$)", min_value=0.0, step=100.0)
            with col_prazo_meta: prazo_meta_meses = st.number_input("Prazo (meses)", min_value=1, step=1)
            prioridade_meta = st.selectbox("Prioridade", ["Alta", "Média", "Baixa"], key="select_prio_meta")
            submit_meta = st.form_submit_button("Adicionar Meta")

            if submit_meta:
                if nome_meta and valor_meta > 0 and prazo_meta_meses > 0:
                    st.session_state.dados_financeiros["metas"][nome_meta] = {
                        "valor": valor_meta, "prazo_meses": prazo_meta_meses, "prioridade": prioridade_meta,
                        "valor_mensal_necessario": valor_meta / prazo_meta_meses, "data_criacao": datetime.now().strftime("%d/%m/%Y")
                    }
                    st.success(f"Meta '{nome_meta}' adicionada!")
                    if len(st.session_state.dados_financeiros["metas"]) == 1 and "Primeira Meta Definida! 🎯" not in st.session_state.conquistas:
                        adicionar_pontos(15, "Definiu sua primeira meta financeira")
                        adicionar_conquista("Primeira Meta Definida! 🎯")
                    # st.rerun()
                else: st.error("Preencha todos os campos da meta.")

        if st.session_state.dados_financeiros["metas"]:
            st.markdown("#### Metas Cadastradas")
            for nome, info in st.session_state.dados_financeiros["metas"].items():
                exp = st.expander(f"{info['prioridade']} - {nome} (R$ {info['valor']:.2f} em {info['prazo_meses']} meses)")
                exp.write(f"Necessário poupar/investir: R$ {info['valor_mensal_necessario']:.2f}/mês")
                exp.caption(f"Criada em: {info['data_criacao']}")
                if exp.button(f"Remover Meta {nome}", key=f"rem_meta_{nome.replace(' ','_')}", type="secondary"):
                    del st.session_state.dados_financeiros["metas"][nome]
                    st.success(f"Meta '{nome}' removida.")
                    st.rerun()
        else: st.caption("Nenhuma meta cadastrada.")

    st.markdown("---")
    if st.button("🏁 Finalizar e Ver Diagnóstico no Dashboard", key="btn_finalizar_diag_total", type="primary", use_container_width=True):
        if not st.session_state.dados_financeiros.get("renda_mensal"):
            st.error("Por favor, informe sua renda mensal na aba 'Renda' para finalizar.")
        else:
            st.session_state.diagnostico_realizado = True
            if "Diagnóstico Completo! 📊" not in st.session_state.conquistas:
                adicionar_pontos(30, "Completou o diagnóstico financeiro")
                adicionar_conquista("Diagnóstico Completo! 📊")
            st.success("✅ Diagnóstico financeiro concluído! Redirecionando para o Dashboard...")
            st.balloons()
            time.sleep(1)
            st.session_state.pagina_atual = "dashboard"
            st.rerun()

# --- PÁGINA DE DESAFIOS (COM CORREÇÃO) ---
def pagina_desafios():
    st.markdown("## 🎯 Desafios Financeiros")
    st.markdown("""
    <div class="info-box">
        <p>Desafios ajudam a desenvolver hábitos financeiros saudáveis de forma divertida. 
        Complete-os para ganhar pontos e melhorar sua saúde financeira!</p>
    </div>
    """, unsafe_allow_html=True)

    # Botão para gerar novo desafio
    # Usar colunas para centralizar o botão
    col_btn_gerar1, col_btn_gerar2, col_btn_gerar3 = st.columns([1,2,1])
    with col_btn_gerar2:
        if st.button("🎲 Gerar Novo Desafio", key="btn_gerar_novo_desafio", use_container_width=True):
            st.session_state.desafio_proposto = gerar_desafio_aleatorio()
            # st.rerun() # Opcional: pode ser útil para limpar outros estados, mas pode causar um piscar.

    # Se um desafio foi proposto e ainda não aceito
    if st.session_state.desafio_proposto:
        desafio = st.session_state.desafio_proposto
        st.markdown("--- \n### ✨ Novo Desafio Proposto ✨")
        # Usar st.container para agrupar o card do desafio proposto
        with st.container():
            st.markdown(f"""
            <div class="challenge-card" style="background-color: #fafafa; border-left: 5px solid #c97ffa;">
                <h4 style="color: #5c1691;">{desafio['titulo']}</h4>
                <p>{desafio['descricao']}</p>
                <p><strong>Dificuldade:</strong> {desafio['dificuldade']} | <strong>Pontos:</strong> {desafio['pontos']}</p>
                <p><strong>Duração:</strong> {desafio['duracao_dias']} dias</p>
            </div>
            """, unsafe_allow_html=True)

            # Botões para aceitar ou recusar o desafio proposto
            # Usar colunas para os botões de aceitar/recusar
            col_aceitar, col_recusar = st.columns(2)
            with col_aceitar:
                # Chave única para o botão de aceitar, baseada no título para evitar conflitos
                if st.button("✅ Aceitar Este Desafio!", key=f"aceitar_desafio_{desafio['titulo'].replace(' ', '_')}", use_container_width=True):
                    aceitar_desafio(st.session_state.desafio_proposto)
                    st.session_state.desafio_proposto = None  # Limpa o desafio proposto após aceitar
                    st.rerun() # Atualiza a UI para mover o desafio para a lista de ativos
            with col_recusar:
                if st.button("❌ Recusar/Gerar Outro", key=f"recusar_desafio_{desafio['titulo'].replace(' ', '_')}", type="secondary", use_container_width=True):
                    st.session_state.desafio_proposto = None # Limpa o desafio proposto
                    st.info("Desafio recusado. Você pode gerar um novo.")
                    st.rerun() # Para limpar o desafio recusado da tela

    st.markdown("---")
    # Exibir desafios ativos
    if st.session_state.desafios_ativos:
        st.markdown("### 👊 Seus Desafios Ativos")
        for i, desafio_ativo in enumerate(st.session_state.desafios_ativos):
            dias_restantes = (desafio_ativo["data_fim"] - datetime.now()).days
            
            with st.expander(f"{desafio_ativo['titulo']} (Restam: {max(0, dias_restantes)} dias | Pontos: {desafio_ativo['pontos']})"):
                st.markdown(f"**Descrição:** {desafio_ativo['descricao']}")
                st.markdown(f"**Dificuldade:** {desafio_ativo['dificuldade']}")
                st.markdown(f"**Início:** {desafio_ativo['data_inicio'].strftime('%d/%m/%Y')} | **Término:** {desafio_ativo['data_fim'].strftime('%d/%m/%Y')}")
                
                # Botões para concluir ou abandonar desafio ativo
                # Usar chaves únicas e distintas das do desafio proposto
                col_btn_concluir, col_btn_abandonar = st.columns(2)
                with col_btn_concluir:
                    if st.button(f"✔️ Marcar como Concluído", key=f"btn_concluir_ativo_{i}_{desafio_ativo['titulo'].replace(' ', '_')}", use_container_width=True):
                        concluir_desafio(i)
                        st.rerun()
                with col_btn_abandonar:
                    if st.button(f"🏳️ Abandonar Desafio", key=f"btn_abandonar_ativo_{i}_{desafio_ativo['titulo'].replace(' ', '_')}", type="secondary", use_container_width=True):
                        titulo_abandonado = st.session_state.desafios_ativos[i]['titulo']
                        st.session_state.desafios_ativos.pop(i)
                        st.warning(f"Desafio '{titulo_abandonado}' abandonado.")
                        # Se o desafio abandonado era o mesmo que estava proposto (caso raro), limpar o proposto.
                        if st.session_state.desafio_proposto and st.session_state.desafio_proposto['titulo'] == titulo_abandonado:
                            st.session_state.desafio_proposto = None
                        st.rerun()
    else:
        st.info("Você não possui desafios ativos no momento. Que tal gerar um novo?")
    
    st.markdown("---")
    # Exibir desafios concluídos
    if st.session_state.desafios_concluidos:
        st.markdown("### 🎉 Desafios Concluídos")
        for desafio_concluido in reversed(st.session_state.desafios_concluidos): # Mostrar mais recentes primeiro
            st.markdown(f"""
            <div class="success-box" style='border-left: 5px solid #4caf50;'>
                <h4>{desafio_concluido['titulo']} ✅</h4>
                <p>{desafio_concluido['descricao']}</p>
                <p><strong>Pontos ganhos:</strong> {desafio_concluido['pontos']}</p>
            </div>
            """, unsafe_allow_html=True)

def pagina_educacional():
    st.markdown("## 📚 Conteúdo Educacional")
    st.markdown("<div class='info-box'><p>Aprenda conceitos financeiros importantes de forma simples e prática para tomar decisões mais inteligentes.</p></div>", unsafe_allow_html=True)
    modulos = [
        {"titulo": "Orçamento Inteligente", "desc": "Domine a arte de criar e seguir um orçamento que funciona para você.", 
         "topicos": ["Registro de Gastos", "Categorização de Despesas", "Regra 50/30/20", "Ferramentas de Orçamento"]},
        {"titulo": "Saindo das Dívidas", "desc": "Estratégias eficazes para se livrar das dívidas e recuperar sua saúde financeira.", 
         "topicos": ["Tipos de Dívidas", "Método Avalanche vs. Bola de Neve", "Negociação com Credores", "Evitando Novas Dívidas"]},
        {"titulo": "Investimentos para Iniciantes", "desc": "Descubra como fazer seu dinheiro trabalhar para você, mesmo começando pequeno.", 
         "topicos": ["Perfil de Investidor", "Renda Fixa (Tesouro Direto, CDB)", "Renda Variável (Ações, FIIs - introdução)", "Diversificação", "Longo Prazo"]},
        {"titulo": "Construindo sua Reserva de Emergência", "desc": "A importância de ter um colchão financeiro e como montá-lo.",
         "topicos": ["O que é e por que ter", "Quanto guardar", "Onde investir a reserva", "Quando usar"]}
    ]
    for i, modulo in enumerate(modulos):
        st.markdown(f"<div class='challenge-card'><h3>{modulo['titulo']}</h3><p>{modulo['desc']}</p></div>", unsafe_allow_html=True)
        with st.expander("Ver Tópicos e Acessar (Simulado)"):
            for topico in modulo["topicos"]: st.markdown(f"- {topico}")
            if st.button(f"Acessar Módulo: {modulo['titulo']}", key=f"btn_mod_{i}_{modulo['titulo'].replace(' ','_')}"):
                st.info(f"Simulação: Acessando conteúdo sobre '{modulo['titulo']}'. Em uma versão completa, aqui teríamos o material detalhado.")
                adicionar_pontos(5, f"Explorou o módulo {modulo['titulo']}")
                if st.button(f"Marcar '{modulo['titulo']}' como Lido", key=f"btn_mod_lido_{i}"):
                    adicionar_pontos(10, f"Concluiu leitura do módulo {modulo['titulo']}")
                    adicionar_conquista(f"Leitura Concluída: {modulo['titulo']}! 📚")
                    st.success(f"Módulo '{modulo['titulo']}' marcado como lido!")
    st.markdown("--- \n ### Dicas Rápidas do Mentor")
    dicas_rapidas = [
        ("Revise Faturas", "Sempre confira suas faturas de cartão e contas para identificar cobranças indevidas ou gastos inesperados."),
        ("Planeje Compras Grandes", "Antes de compras caras, pesquise, compare preços e veja se cabe no orçamento. Evite o impulso!"),
        ("Converse sobre Dinheiro", "Falar sobre finanças com parceiro(a) ou família ajuda a alinhar objetivos e evitar conflitos.")
    ]
    cols_dicas = st.columns(len(dicas_rapidas))
    for idx, (titulo_dica, desc_dica) in enumerate(dicas_rapidas):
        with cols_dicas[idx]:
            st.markdown(f"<div class='info-box' style='background-color: #e3f2fd; border-left-color: #2196f3;'><h4>{titulo_dica}</h4><p>{desc_dica}</p></div>", unsafe_allow_html=True)

def pagina_conquistas():
    st.markdown("## 🏆 Suas Conquistas e Progresso")
    col_nivel, col_medalha = st.columns([2,1])
    with col_nivel:
        st.markdown(f"### Nível Atual: {st.session_state.nivel}")
        progresso_nivel_val = (st.session_state.pontos % 100)
        st.progress(progresso_nivel_val / 100, text=f"{progresso_nivel_val}/100 pontos para o Nível {st.session_state.nivel + 1}")
        st.markdown(f"**Pontos Totais:** {st.session_state.pontos}")
    with col_medalha:
        medalha_simbolo, medalha_nome = "", ""
        if st.session_state.nivel >= 5: medalha_simbolo, medalha_nome = "🏆", "Especialista Financeiro"
        elif st.session_state.nivel >= 3: medalha_simbolo, medalha_nome = "🥈", "Estrategista Financeiro"
        else: medalha_simbolo, medalha_nome = "🥉", "Aprendiz Financeiro"
        st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 3rem; margin-bottom:0;'>{medalha_simbolo}</h1><p style='font-weight: bold; color: #5c1691;'>{medalha_nome}</p></div>", unsafe_allow_html=True)

    st.markdown("--- \n### Suas Medalhas de Honra (Conquistas)")
    if st.session_state.conquistas:
        cols_conq = st.columns(3) # 3 colunas para conquistas
        for i, conquista_item in enumerate(st.session_state.conquistas):
            with cols_conq[i % 3]:
                st.markdown(f"<div class='success-box' style='text-align:center; padding: 15px;'><p style='font-weight:bold; margin-bottom:5px;'>{conquista_item.split('!')[0]}!</p><p style='font-size: 2rem;'>{conquista_item.split(' ')[-1]}</p></div>", unsafe_allow_html=True)
    else: st.info("Continue usando o app para desbloquear novas conquistas!")
    
    st.markdown("--- \n### Estatísticas Gerais")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1: st.metric("Consultas ao Mentor AI", len(st.session_state.historico_consultas))
    with col_stats2: st.metric("Desafios Concluídos", len(st.session_state.desafios_concluidos))
    with col_stats3: st.metric("Metas Definidas", len(st.session_state.dados_financeiros.get("metas", {})))
    
    st.markdown("--- \n### Próximas Conquistas Sugeridas")
    sugestoes_conq = [
        "Complete 3 desafios de dificuldade 'Média' ou 'Difícil' 🎯",
        "Atinja o Nível 5 de experiência 🌟",
        "Mantenha o comprometimento de renda abaixo de 60% por um mês 💰",
        "Construa uma reserva de emergência para 1 mês de despesas 🛡️",
        "Aprenda 5 novos termos no glossário financeiro 📚"
    ]
    for sug_conq in sugestoes_conq:
        ja_tem = any(sug_conq_base in c_atual for c_atual in st.session_state.conquistas for sug_conq_base in [sug_conq.split(' ')[-2]]) # Heurística simples
        if not ja_tem: st.markdown(f"- {sug_conq}")

# --- Função Principal ---
def main():
    inicializar_sessao()
    exibir_cabecalho() # Exibe antes da sidebar para consistência
    exibir_barra_lateral() # A navegação aqui pode chamar st.rerun()
    
    # Roteamento de páginas
    if st.session_state.pagina_atual == "boas_vindas": pagina_boas_vindas()
    elif st.session_state.pagina_atual == "dashboard": pagina_dashboard()
    elif st.session_state.pagina_atual == "consultor": pagina_consultor()
    elif st.session_state.pagina_atual == "diagnostico": pagina_diagnostico()
    elif st.session_state.pagina_atual == "desafios": pagina_desafios()
    elif st.session_state.pagina_atual == "educacional": pagina_educacional()
    elif st.session_state.pagina_atual == "conquistas": pagina_conquistas()
    else: st.session_state.pagina_atual = "boas_vindas"; st.rerun() # Fallback

if __name__ == "__main__":
    main()
