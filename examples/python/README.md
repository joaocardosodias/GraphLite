# Exemplo Python: Download da Internet, Ingestao e Consulta do Codigo Penal com Graphite DB

Este exemplo demonstra como utilizar o SDK Python oficial do Graphite DB (`graphite-database`) para:
1. **Baixar diretamente da internet** o documento completo do Codigo Penal em formato Markdown (`.md`).
2. **Ingerir e vetorizar** cada artigo e dispositivo legal gerando embeddings de 384 dimensoes localmente na CPU com FastEmbed.
3. **Construir um Grafo de Conhecimento** conectando tipos de crimes, excludentes de ilicitude e relacoes juridicas.
4. **Executar consultas GraphRAG** em linguagem natural com sintese de contexto e respeito estrito ao orcamento de tokens.

---

## Como Executar

### 1. Instalar as dependencias
```bash
pip install graphite-database
```

### 2. Executar o script
```bash
python main.py
```

---

## Estrutura dos Arquivos

* `main.py` — Script principal que realiza o download HTTP via urllib, parsing hierarquico, ingestao no Graphite e execucao das perguntas.
* `codigo_penal.md` — Documento Markdown de referencia com 186 linhas contendo os principais artigos da Parte Geral e Especial do Codigo Penal (Decreto-Lei No 2.848/1940).
