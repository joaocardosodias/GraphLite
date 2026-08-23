# GraphLite Examples & Document Knowledge Base

Esta pasta contém exemplos práticos de documentos em múltiplos formatos para você testar a ingestão, o grafo de conhecimento e a recuperação de contexto para agentes de IA e chatbots.

---

## 📁 Documentos de Exemplo (`examples/documents/`)

1. **`politica_reembolso_e_cancelamento.md`**: Regras de cancelamento, direito de arrependimento (7 dias) e estornos via PIX/Cartão.
2. **`manual_lgpd_seguranca.md`**: Diretrizes de privacidade, direitos dos titulares e contato do DPO (Dra. Mariana Siqueira).
3. **`faq_suporte_tecnico.txt`**: Perguntas frequentes sobre reset de senha, 2FA/MFA e rate limits de API.
4. **`catalogo_planos_enterprise.json`**: Estrutura JSON com planos de assinatura (Starter, Business, Enterprise).
5. **`tabela_precos_servicos.csv`**: Tabela CSV com catálogo de serviços profissionais e preços.
6. **`guia_arquitetura_api.md`**: Especificação técnica de autenticação OAuth 2.0 JWT e Webhooks com assinatura HMAC-SHA256.

---

## 🚀 Como Executar

### 1. Ingerir a Pasta de Documentos:

```bash
graphlite -d examples/knowledge.graph ingest examples/documents
```

### 2. Registrar Memórias e Preferências de Agentes:

```bash
graphlite -d examples/knowledge.graph remember \
  "O cliente Dr. Marcos da Silva é diretor de TI, usa o plano Enterprise Custom e exige reuniões apenas às terças-feiras via Google Meet" \
  --category "UserPreference"

graphlite -d examples/knowledge.graph remember \
  "Regra interna de compliance: Qualquer exportação de dados de usuários acima de 1.000 registros requer aprovação formal da DPO Mariana Siqueira" \
  --category "ComplianceRule"
```

### 3. Consultar o Grafo de Conhecimento (GraphRAG):

```bash
# Pergunta sobre reembolso e estorno via PIX:
graphlite -d examples/knowledge.graph query -T "qual o prazo para pedir reembolso integral e como funciona via PIX?"

# Pergunta sobre segurança e compliance:
graphlite -d examples/knowledge.graph query -T "quem é o DPO da empresa e qual a regra para exportar dados?"

# Pergunta sobre planos e preferências do usuário:
graphlite -d examples/knowledge.graph query -T "quais os benefícios do plano Enterprise e preferências do Dr Marcos?"
```
