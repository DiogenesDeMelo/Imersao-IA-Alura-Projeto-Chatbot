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

    # Gerar novo desafio (armazenar na sessão para persistência)
    if 'novo_desafio' not in st.session_state:
        st.session_state.novo_desafio = None

    if st.button("Gerar Novo Desafio", key="btn_novo_desafio"):
        st.session_state.novo_desafio = gerar_desafio_aleatorio()

    # Exibir novo desafio gerado
    if st.session_state.novo_desafio is not None:
        desafio = st.session_state.novo_desafio
        with st.container():
            st.markdown(f"""
            <div class="challenge-card">
                <h3>{desafio['titulo']}</h3>
                <p>{desafio['descricao']}</p>
                <p><strong>Dificuldade:</strong> {desafio['dificuldade']} | <strong>Pontos:</strong> {desafio['pontos']}</p>
                <p><strong>Duração:</strong> {desafio['duracao_dias']} dias</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Aceitar Desafio", key=f"aceitar_{desafio['titulo']}"):
                aceitar_desafio(desafio)
                st.session_state.novo_desafio = None  # Limpar desafio gerado
                st.rerun()

    # Exibir desafios ativos com gerenciamento
    if st.session_state.desafios_ativos:
        st.markdown("### Desafios Ativos")
        
        for i, desafio in enumerate(st.session_state.desafios_ativos[:]):  # Iterar sobre cópia para evitar problemas de índice
            dias_restantes = (desafio["data_fim"] - datetime.now()).days
            expander = st.expander(f"{desafio['titulo']} ({max(0, dias_restantes)} dias restantes)")
            
            with expander:
                st.markdown(f"**Descrição:** {desafio['descricao']}")
                st.markdown(f"**Dificuldade:** {desafio['dificuldade']}")
                st.markdown(f"**Pontos:** {desafio['pontos']}")
                st.markdown(f"**Data de Início:** {desafio['data_inicio'].strftime('%d/%m/%Y')}")
                st.markdown(f"**Data de Término:** {desafio['data_fim'].strftime('%d/%m/%Y')}")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button(f"✅ Concluir", key=f"concluir_{i}"):
                        concluir_desafio(i)
                        st.rerun()
                with col2:
                    if st.button(f"❌ Abandonar", key=f"abandonar_{i}"):
                        st.session_state.desafios_ativos.pop(i)
                        st.warning("Desafio abandonado.")
                        st.rerun()

    # Exibir desafios concluídos
    if st.session_state.desafios_concluidos:
        st.markdown("### Histórico de Desafios")
        for desafio in st.session_state.desafios_concluidos:
            st.markdown(f"""
            <div class="success-box">
                <h4>{desafio['titulo']} ✅</h4>
                <p>{desafio['descricao']}</p>
                <p><strong>Pontos ganhos:</strong> {desafio['pontos']}</p>
            </div>
            """, unsafe_allow_html=True)
