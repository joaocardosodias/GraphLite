# Guia de Arquitetura e Integração de APIs

## 1. Padrões de Autenticação
A plataforma adota o padrão **OAuth 2.0 com Bearer Tokens JWT (JSON Web Tokens)**:
- O cabeçalho de autenticação deve ser enviado no formato: `Authorization: Bearer <seu_token_aqui>`.
- O tempo de expiração do access token é de **60 minutos**.
- Para renovar o token, utilize o endpoint `/api/v1/auth/refresh` com seu refresh token de longa duração (30 dias).

## 2. Webhooks e Notificações Assíncronas
- Eventos de pagamento aprovado, cancelamento de plano e alertas de segurança são emitidos via Webhooks HTTPS POST.
- Cada requisição de webhook inclui a assinatura `X-Signature-SHA256` no cabeçalho para validação de autenticidade contra ataques de falsificação.
- Política de Retry: tentativas automáticas de entrega com backoff exponencial nos intervalos de 1min, 5min, 30min e 2h.
