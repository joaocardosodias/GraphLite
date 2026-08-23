# Diretrizes de Memória e Conhecimento do Projeto

Este projeto utiliza o **GraphLite** como motor de memória arquitetural e grafo de conhecimento persistido em `graphlite.graph`.

## Regras Obrigatórias para Assistentes de IA:
1. **Consulta de Arquitetura e Regras:**
- Sempre que o usuário fizer perguntas sobre como o sistema funciona, regras de negócio, autenticação, modelos de dados ou endpoints, você **DEVE** acionar a ferramenta MCP `graphlite_retrieve(query: "...")`.
2. **Proibição de Leitura Direta de Binários:**
- O arquivo `graphlite.graph` é um banco binário compilado. **NUNCA** tente ler `graphlite.graph` usando `cat`, `read_file`, `head` ou utilitários SQL como `sqlite3`.
- Todas as operações de leitura devem ser feitas exclusivamente via `graphlite_retrieve`.
3. **Persistência de Novas Regras:**
- Ao aprender novas regras ou preferências do usuário, chame `graphlite_remember` e `graphlite_connect`.
