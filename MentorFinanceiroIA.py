import google.generativeai as genai
import os
import time
import json
import streamlit as st

# --- Configuração da API Key ---
try:
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "Variável de ambiente GOOGLE_API_KEY não encontrada. Por favor, configure-a no arquivo secrets.toml do Streamlit."
        )
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Erro ao configurar a API Key: {e}")
    st.error(
        "Verifique se você configurou a variável de ambiente GOOGLE_API_KEY corretamente."
    )
    st.stop()

# --- Configuração do Modelo Generativo ---
generation_config = {
    "temperature": 0.75,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

try:
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
except Exception as e:
    st.error(f"Erro ao inicializar o modelo Gemini: {e}")
    st.stop()

# --- Definição das Conquistas ---
conquistas = {
    5: "🏆Novato Financeiro🏆",
    10: "🏆Aprendiz das Finanças🏆",
    15: "🏆Entusiasta Financeiro🏆",
    20: "🏆Mestre das Finanças🏆",
    25: "🏆Guru Financeiro🏆",
    30: "🏆Oráculo Financeiro🏆",
}


# --- Funções Auxiliares ---
def exibir_mensagem_boas_vindas():
    """Exibe uma mensagem de boas-vindas ao usuário."""
    st.title("🌟 Bem-vindo ao Mentor Financeiro AI! 🌟")
    st.write("🕵️ Seu assistente para trilhar o caminho da saúde financeira.")
    st.write(
        "🎉 Ao longo dessa jornada, quanto mais você perguntar, maior será sua pontuação!."
    )
    st.write(
        "🏆 Quanto mais insights financeiros você obtiver, você irá melhorar a sua saúde financeira e ainda obter mais conhecimento e desbloquear conquistas!"
    )



def coletar_dados_usuario(nome_usuario_ja_coletado=None):
    """Coleta informações básicas e financeiras do usuário."""
    nome = nome_usuario_ja_coletado or st.session_state.nome if nome_usuario_ja_coletado or 'nome' in st.session_state else st.text_input("Olá! Para começarmos, qual o seu nome?").strip()
    if not nome:
        st.warning(
            "Hummmm, você não informou seu nome! Para começarmos nossa jornada, insira o seu nome!."
        )
        st.stop()
    st.write(f"\nPrazer em te conhecer, {nome}! Fico muito feliz em ter você por aqui!")
    st.session_state.nome = nome  # Salva o nome no session_state

    preocupacao = st.text_input(
        f"\n{nome}, vamos começar com o seguinte: Qual sua principal preocupação financeira ou dívida no momento? \n"
        "(Ex: 'cartão de crédito com R$2000', 'não consigo guardar dinheiro', 'quero renegociar meu financiamento')", key="preocupacao"
    ).strip()
    if not preocupacao:
        st.warning(
            "Não fique acanhado, conta comigo nesse processo! Agora, descreva sua preocupação para que eu possa te ajudar."
        )
        st.stop()
    st.session_state.preocupacao = preocupacao

    renda_mensal_str = st.text_input(
        f"\nPara te ajudar melhor, qual sua renda mensal aproximada? (Pressione Enter se preferir não informar ou sua preocupação não precise dessa informação): R$ ", key="renda_mensal"
    ).strip()
    renda_mensal = None
    if renda_mensal_str:
        try:
            renda_mensal = float(renda_mensal_str.replace(",", "."))
        except ValueError:
            st.error("Valor de renda inválido, seguirei sem essa informação.")
            renda_mensal = None
    st.session_state.renda_mensal = renda_mensal

    despesa_mensal_str = st.text_input(
        f"\nSe preferir, Poderia informar sua despesa mensal total estimada? (Pressione Enter se preferir não informar ou sua preocupação não precise dessa informação): R$ ", key="despesa_mensal"
    ).strip()
    despesa_mensal = None
    if despesa_mensal_str:
        try:
            despesa_mensal = float(despesa_mensal_str.replace(",", "."))
        except ValueError:
            st.error("Valor de despesa inválido, seguirei sem essa informação.")
            despesa_mensal = None
    st.session_state.despesa_mensal = despesa_mensal

    valor_divida_str = st.text_input(
        f"\nSe sua preocupação é uma dívida específica, ou se você tem um montate total de dívidas, qual o valor aproximado dela(s)? (Pressione Enter se não aplicável ou não quiser informar): R$ ", key="valor_divida"
    ).strip()
    valor_divida = None
    if valor_divida_str:
        try:
            valor_divida = float(valor_divida_str.replace(",", "."))
        except ValueError:
            st.error("Valor de dívida inválido, seguirei sem essa informação.")
            valor_divida = None
    st.session_state.valor_divida = valor_divida
    return nome, preocupacao, renda_mensal, despesa_mensal, valor_divida



def gerar_conselho_financeiro_avancado(
    nome_usuario,
    preocupacao_usuario,
    renda_mensal=None,
    despesa_mensal=None,
    valor_divida=None,
):
    """
    Usa a API Gemini para gerar um conselho financeiro mais elaborado.
    """
    st.write("\n🧠 Aguarde enquanto O Mentor Financeiro IA processa suas informações...")
    time.sleep(1)

    prompt_parts = [
        f"Meu nome é {nome_usuario}.",
        f"Minha principal preocupação financeira é: '{preocupacao_usuario}'.",
    ]
    if renda_mensal is not None:
        prompt_parts.append(f"Minha renda mensal aproximada é de R${renda_mensal:.2f}.")
    if despesa_mensal is not None:
        prompt_parts.append(f"Minha despesa mensal aproximada é de R${despesa_mensal:.2f}.")
    if valor_divida is not None:
        prompt_parts.append(f"O valor aproximado dessa dívida/preocupação é de R${valor_divida:.2f}.")

    prompt_parts.extend([
        "\nVocê é um consultor financeiro experiente, empático, motivador e detalhista.",
        "Preciso de ajuda para lidar com essa situação descrita.",
        "Forneça para mim, em português do Brasil:",
        "1. Uma mensagem curta de encorajamento e validação dos meus sentimentos (1-2 frases).",
        "2. Caso eu tenha informa minha renda, despesa e dívida, elabore um plano financeiro detalhado, de forma a me propor soluções bem estruturadas de como corrigir minha situação financeira.",
        "3. Caso minha renda não seja suficiente para traçar um plano, me sugira forma de conseguir uma renda extra, para conseguir ajustar a saúde financeira",
        "4. Uma dica extra ou uma reflexão positiva curta (1 frase).",
        "Seja claro, direto e use uma linguagem acessível. Evite jargões financeiros complexos.",
    ])

    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"\n❌ Desculpe, ocorreu um erro ao tentar gerar o conselho: {e}")
        st.error(
            "Por favor, verifique sua conexão com a internet e a configuração da API Key."
        )
        return "Não foi possível gerar um conselho no momento. Tente novamente mais tarde."



def exibir_conselho(nome_usuario, conselho, contador_consultas):
    """Exibe o conselho gerado pela IA."""
    st.write("-" * 60)
    st.write(
        f"💡 {nome_usuario}, aqui está a orientação #{contador_consultas} do Mentor Financeiro AI:\n"
    )
    st.write(conselho)
    st.write("-" * 60)
    if contador_consultas == 1:
        st.write("\n✨ Este é o seu primeiro insight! Continue buscando conhecimento e agindo!")
        st.write(
            "\n✨ Lembre-se, esse insight é gerador por IA, que pode cometer erros! Sempre busque orientação profissional adequada!"
        )
    else:
        st.write(
            f"\n✨ Você já obteve {contador_consultas} insights! Continue sua jornada de aprendizado e ação!"
        )
    st.write("Dica: pequenas ações consistentes levam a grandes resultados.")



def salvar_progresso(nome_usuario, contador_consultas):
    """Salva o progresso do usuário em um arquivo JSON."""
    dados = {"nome_usuario": nome_usuario, "contador_consultas": contador_consultas}
    try:
        with open(f"{nome_usuario}_progresso.json", "w") as arquivo:
            json.dump(dados, arquivo, indent=4)
    except Exception as e:
        st.error(f"Erro ao salvar o progresso: {e}")



def carregar_progresso():
    """Carrega o progresso do usuário de um arquivo JSON."""
    try:
        # Tenta abrir o arquivo sem depender do nome do usuário
        arquivos_progresso = [
            f
            for f in os.listdir()
            if f.endswith("_progresso.json")
        ]  # Lista arquivos

        if arquivos_progresso:
            # Se encontrar algum arquivo, pega o primeiro
            nome_arquivo = arquivos_progresso[0]
            with open(nome_arquivo, "r") as arquivo:
                dados = json.load(arquivo)
                if (
                    "nome_usuario" in dados and "contador_consultas" in dados
                ):  # Verifica se as chaves existem
                    nome_usuario = dados["nome_usuario"]
                    contador_consultas = dados["contador_consultas"]
                    st.write(f"Progresso carregado de: {nome_arquivo}")
                    return nome_usuario, contador_consultas
                else:
                    st.write(
                        "Arquivo de progresso corrompido. Iniciando novo progresso."
                    )  # Informa
                    return None, 0
        else:
            st.write("Nenhum progresso anterior encontrado. Iniciando novo progresso.")
            return None, 0  # Retorna None para o nome se não achar
    except json.JSONDecodeError:
        st.write(
            "Erro ao decodificar o arquivo de progresso. Iniciando novo progresso."
        )  # Trata
        return None, 0
    except Exception as e:
        st.error(f"Erro ao carregar o progresso: {e}")
        return None, 0


def main():
    """Função principal do programa."""
    exibir_mensagem_boas_vindas()
    nome_usuario_cache = None
    contador_consultas = 0

    # Carrega o progresso.
    nome_usuario_cache, contador_consultas = carregar_progresso()

    # Se não conseguiu carregar o nome do usuário, coleta do usuário.
    if nome_usuario_cache is None:
        nome_usuario_cache, _, _, _, _ = coletar_dados_usuario()
    else:
        st.write(
            f"Olá novamente, {nome_usuario_cache}! Vamos continuar em busca de uma vida financeira mais saudável e próspera!!")

    # Usar st.session_state para manter o estado entre as iterações
    if 'contador_consultas' not in st.session_state:
        st.session_state.contador_consultas = contador_consultas
    if 'nome' not in st.session_state:
        st.session_state.nome = nome_usuario_cache
    nome_usuario = st.session_state.nome #garantir que nome_usuario está definido


    while True:
        # Passa nome_usuario_cache para coletar_dados_usuario em cada iteração
        nome_usuario, preocupacao, renda_mensal, despesa_mensal, valor_divida = coletar_dados_usuario(nome_usuario_ja_coletado=nome_usuario)

        st.session_state.contador_consultas += 1
        conselho = gerar_conselho_financeiro_avancado(
            nome_usuario, preocupacao, renda_mensal, despesa_mensal, valor_divida
        )
        exibir_conselho(nome_usuario, conselho, st.session_state.contador_consultas)
        salvar_progresso(
            nome_usuario, st.session_state.contador_consultas
        )  # Salva o progresso do usuário

        # --- Verificação de Conquistas ---
        if st.session_state.contador_consultas in conquistas:
            st.write("-" * 60)
            st.write(
                f"🏆 Parabéns, {nome_usuario}! Você desbloqueou a conquista: {conquistas[st.session_state.contador_consultas]} 🏆"
            )
            st.write("-" * 60)

        time.sleep(1)
        continuar = st.radio(
            "\nDeseja fazer outra consulta ou registrar outra preocupação?",
            ["Sim", "Não"], key="continuar"
        )
        if continuar == "Não":
            st.write(
                f"\nObrigado por usar o Mentor Financeiro AI, {nome_usuario}! Volte sempre que precisar de um norte! 🚀"
            )
            st.write("Boa sorte no seu desafio da Alura! Você está no caminho certo!")
            break
        st.write("\nOk! Vamos para a próxima consulta...")
        time.sleep(1)


if __name__ == "__main__":
    main()
