# GraphLite Examples & Document Knowledge Base

Esta pasta contém exemplos práticos de documentos em múltiplos formatos para você testar a ingestão, o grafo de conhecimento e a recuperação de contexto para agentes de IA e chatbots.

---

## 📁 Documentos Reais Baixados da Internet (`examples/documents/`)

A base contém **documentos autênticos e oficiais baixados diretamente da internet** em múltiplos formatos (.pdf, .md, .txt, .yaml, .json, .csv):

1. **`attention_is_all_you_need.pdf`** (~2.2 MB): Artigo acadêmico original do arXiv sobre a arquitetura de Transformers e mecanismos de atenção (*Vaswani et al.*).
2. **`rfc9110_http_semantics.txt`** (~492 KB): Especificação técnica oficial do IETF sobre Semântica e Códigos de Status HTTP (200, 404, 500).
3. **`rfc7519_jwt_standard.txt`** (~62 KB): Especificação técnica oficial do IETF para JSON Web Tokens (JWT Claims: `iss`, `sub`, `exp`).
4. **`tokio_async_runtime.md`** (~9.1 KB): Guia oficial e arquitetura de concorrência e tasks assíncronas do runtime Tokio Rust.
5. **`rust_api_guidelines.md`** (~7.1 KB): Checklist oficial de design e convenções de nomenclatura de APIs em Rust.
6. **`rust_book_concurrency.md`** & **`rust_book_cli_project.md`** (~5.5 KB): Capítulos oficiais do Rust Lang Book sobre concorrência sem medo e criação de CLIs.
7. **`google_microservices_k8s.yaml`** (~23 KB): Manifestos de produção do Google Cloud Platform para arquitetura de microsserviços em Kubernetes.
8. **`prometheus_node_exporter_values.yaml`** (~22 KB): Helm Chart de métricas de infraestrutura do Prometheus Community.
9. **`nobel_prizes_archive.json`** (~228 KB): Dataset da API oficial da Fundação Nobel com todos os prêmios da história.
10. **`global_countries_dataset.json`** (~1.4 MB): Base de dados geográficos com todos os países, moedas e capitais do mundo.
11. **`titanic_passengers_dataset.csv`** (~59 KB): Dataset do Kaggle / DataScienceDojo sobre passageiros e sobrevivência do Titanic.
12. **`exoplanets_astronomy.csv`** (~36 KB): Base de dados astronômicos de exoplanetas descobertos pela NASA / Seaborn.
13. **`world_gdp_2014.csv`** & **`global_flights_dataset.csv`** (~6.8 KB): Dados econômicos do Banco Mundial e histórico de voos.
14. **`w3c_sample_accessibility_doc.pdf`** (~13 KB): Documento técnico de teste de acessibilidade do W3C.
15. **Manuais e Contratos Corporativos:** SLA empresarial, Manual de Engenharia, Auditoria SOC 2 e LGPD.

---

## 🚀 Como Executar

### 1. Ingerir a Pasta de Documentos com Caching Incremental:

```bash
graphlite -d examples/knowledge.graph ingest examples/documents
```

### 2. Iniciar o Servidor REST API Local:

```bash
graphlite -d examples/knowledge.graph serve --port 8000
```

### 3. Consultar o Grafo de Conhecimento (GraphRAG):

```bash
# Pergunta sobre SLA contratual e créditos por indisponibilidade:
graphlite -d examples/knowledge.graph query -T "qual o SLA de Uptime garantido e quais as penalidades se ficar fora do ar?"

# Pergunta sobre regras de deploy e aprovação de PR:
graphlite -d examples/knowledge.graph query -T "quais as regras para aprovar PR e fazer deploy canary no kubernetes?"

# Pergunta sobre estoque e hardware:
graphlite -d examples/knowledge.graph query -T "quais os servidores edge appliance disponiveis no estoque e os precos?"

# Pergunta sobre auditoria SOC 2 e segurança:
graphlite -d examples/knowledge.graph query -T "quais os padroes de criptografia validados na auditoria SOC2?"
```
