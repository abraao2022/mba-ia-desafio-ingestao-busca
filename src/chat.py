from search import search_prompt

def print_header():
    """Exibe o cabeçalho do chat."""
    print("\n" + "="*60)
    print("🤖 CHAT RAG - Sistema de Perguntas e Respostas")
    print("="*60)
    print("📚 Faça perguntas sobre o documento ingerido")
    print("💡 Digite 'sair', 'exit' ou 'quit' para encerrar")
    print("="*60 + "\n")

def main():
    """Função principal que gerencia o loop de chat interativo."""
    print_header()

    # Verifica se há argumentos de linha de comando para pergunta única
    import sys
    if len(sys.argv) > 1:
        # Modo de pergunta única via argumento
        pergunta = " ".join(sys.argv[1:])
        print(f"Modo de pergunta única: {pergunta}\n")
        resultado = search_prompt(pergunta)
        return

    # Loop interativo de chat
    while True:
        try:
            # Solicita entrada do usuário
            pergunta = input("\n💬 Você: ").strip()

            # Verifica comandos de saída
            if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Encerrando o chat. Até logo!")
                break

            # Verifica se a pergunta está vazia
            if not pergunta:
                print("⚠️  Por favor, digite uma pergunta válida.")
                continue

            # Processa a pergunta usando o search_prompt
            resultado = search_prompt(pergunta)

            if not resultado:
                print("❌ Não foi possível obter uma resposta. Tente novamente.")

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro durante o processamento: {e}")
            print("Tente novamente ou digite 'sair' para encerrar.")

if __name__ == "__main__":
    main()