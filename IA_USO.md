# Registro de uso de IA generativa

O uso de IA generativa neste projeto foi registrado conforme exigido no enunciado (seção 3).

## Uso inicial (versão original do grupo)
- Esclarecimento de conceitos de complexidade assintótica (Big O) e de métricas de caixa/margem.
- Revisão de sintaxe e depuração pontual (tratamento de exceções, estruturas NumPy/Pandas).

## Revisão técnica assistida por IA (30/10/2026)
Uma IA (Claude, Anthropic) foi usada para **auditar** a entrega contra o enunciado e **reescrever/corrigir** partes do código:
- Q2: o código original lia colunas/chaves inexistentes (usava valores padrão no lugar das dívidas, estoques e demanda do grupo), ignorava estoque na validação e o caixa inicial; foi reescrito (`questao_02/`).
- Q1: chaves de configuração corrigidas, GA vetorizado, Greedy com ganho líquido (o original tinha ganho R$ 0,00), benchmark e testes.
- Geração de gráficos, benchmarks, testes e rascunho da documentação em `docs/`.

**Responsabilidade do grupo:** os integrantes devem ler, entender e ser capazes de explicar e alterar qualquer trecho (a defesa pode sortear qualquer função). As premissas de modelagem que **não vêm da instância** estão listadas em `docs/decisoes.md` (seção "Premissas próprias") e podem ser alteradas por parâmetro.
