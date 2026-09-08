# Plano de Entidades VHDL — Inferência de CNN em FPGA

## 1. Contexto do modelo

Rede treinada em Keras/MNIST:

| Camada | Operação | Entrada | Saída |
|---|---|---|---|
| Conv1 | Conv2D 32 filtros 5x5, ReLU | 28x28x1 | 24x24x32 |
| Pool1 | MaxPool 2x2 | 24x24x32 | 12x12x32 |
| Conv2 | Conv2D 64 filtros 5x5, ReLU | 12x12x32 | 8x8x64 |
| Pool2 | MaxPool 2x2 | 8x8x64 | 4x4x64 |
| Flatten | — | 4x4x64 | 1024 |
| Dense | 10 neurônios, Softmax | 1024 | 10 |

Pesos e bias já exportados em INT8 (quantização simétrica) via `.mem` (hex) + arquivo de escala (`scale`) por camada.

## 2. Decisões de projeto (definir ANTES de escrever qualquer entity)

Essas decisões mudam a interface de quase todas as entities, então precisam ser fechadas primeiro:

1. **Acumulador**: MAC INT8 x INT8 estoura rápido — use acumulador INT32 internamente em cada camada.
2. **Requantização — FECHADA**: usa multiplicador inteiro + shift fixo (`SHIFT = 16` para todo o projeto), no formato `((acc_int32 * MULTIPLIER) + arredondamento) >> SHIFT`, com saturação em `[-127, 127]`. `MULTIPLIER` é calculado em Python a partir de `(scale_entrada * scale_peso) / scale_saída` (essa última vem de calibração — ver item 6) e é uma **constante por camada**, já exportada em `conv1_multiplier.txt`, `conv2_multiplier.txt` e `dense_multiplier.txt`. A função Python `requantize_int32_to_int8` faz exatamente essa conta e deve ser usada para gerar os valores de referência dos testbenches (item 1 da seção 5).
3. **Softmax → Argmax**: softmax é monótono, então para inferência (não treino) o índice de maior valor é o mesmo antes e depois do softmax. Sugestão: substituir a saída por um bloco de **argmax**, muito mais barato em hardware do que exponencial + divisão.
4. **Paralelismo dos MACs**: decidir se cada filtro da convolução roda em paralelo (rápido, gasta mais DSP/LUT) ou se você reusa poucos MACs de forma sequencial (lento, gasta pouco recurso). Para começar, recomendo **sequencial** — é mais simples de depurar e cabe em qualquer FPGA pequena.
5. **Armazenamento entre camadas**: os mapas de características intermediários (ex: 24x24x32 = 18.432 valores) não cabem em registradores — vão precisar de BRAM (memória interna do FPGA), então cada camada terá uma saída em uma RAM dual-port que a próxima camada lê.
6. **Escala de saída de cada camada — FECHADA**: obtida por calibração em Python (rodar ~200 imagens de exemplo pelo modelo e pegar o maior valor absoluto de ativação de cada camada). Simplificações já aplicadas: o MaxPool reusa a escala de saída da convolução anterior (não precisa de calibração própria, já que só escolhe um valor existente); e a camada Dense final não precisa de requantização nenhuma se a saída for por **argmax** — o índice do maior valor no acumulador INT32 "cru" já é a resposta.

## 3. Hierarquia de entidades

### Nível 0 — Blocos básicos (primitivos)
- **`mac_int8`** — multiplica dois INT8 e acumula em INT32. Base de tudo.
- **`relu_unit`** — zera valores negativos.
- **`requant_unit`** — recebe o acumulador INT32, multiplica por uma constante `MULTIPLIER` (generic, um valor por camada — vem de `*_multiplier.txt`), soma o arredondamento, desloca `SHIFT` bits à direita e satura em `[-127, 127]`. Interface fechada — ver item 2 da seção 2.
- **`comparator_max2`** — compara dois valores INT8, retorna o maior. Base do max pooling.

### Nível 1 — Memória e movimentação de dados
- **`weight_rom`** — ROM genérica que carrega um `.mem`. Será instanciada uma vez por conjunto de pesos/bias (conv1_weights, conv1_bias, conv2_weights, conv2_bias, dense_weights, dense_bias). O `MULTIPLIER`/`SHIFT` de cada camada **não** passa por aqui — é um valor único (não um array), então entra como `generic`/`constant` direto na `requant_unit`, lido do `*_multiplier.txt` na hora de gerar o pacote de constantes VHDL.
- **`line_buffer`** — shift-register que guarda N linhas da imagem/mapa de características, necessário pra gerar a janela deslizante da convolução.
- **`window_gen`** — usa o `line_buffer` pra entregar, a cada ciclo, uma janela 5x5 (convolução) ou 2x2 (pooling).
- **`feature_map_ram`** — RAM dual-port que guarda a saída de uma camada até a próxima estar pronta pra consumir.

### Nível 2 — Camadas completas (cada uma integra os blocos acima)
- **`conv_layer`** (genérica: tamanho de imagem, kernel, canais entrada/saída) — junta `window_gen` + `mac_int8` + soma de bias + `requant_unit` + `relu_unit`.
- **`maxpool_layer`** (genérica: tamanho de imagem, canais, tamanho da janela) — junta `window_gen` (2x2) + `comparator_max2`.
- **`dense_layer`** (genérica: nº de entradas, nº de saídas) — MAC sequencial sobre o vetor achatado, soma de bias, `requant_unit`.
- **`argmax_unit`** — recebe os 10 valores de saída, entrega o índice do maior.

### Nível 3 — Integração
- **`control_fsm`** — máquina de estados que sequencia as camadas (IDLE → CARREGA_IMAGEM → CONV1 → POOL1 → CONV2 → POOL2 → DENSE → ARGMAX → PRONTO), disparando `start` e aguardando `done` de cada camada.
- **`cnn_top`** — entity de topo, instancia todas as camadas + `control_fsm`, expõe as portas externas (entrada da imagem, saída da classe prevista, clock/reset).

### Nível 4 — Verificação (não é hardware, mas faz parte do plano)
- Um testbench por entity (`tb_mac_int8`, `tb_conv_layer`, ..., `tb_cnn_top`), comparando com valores de referência gerados a partir do seu modelo Python.

## 4. Ordem de implementação recomendada

Construa de baixo pra cima, validando cada peça isoladamente antes de montar a próxima — é bem mais fácil achar bug em uma peça pequena do que no sistema inteiro.

1. **`mac_int8`** + testbench — confirme que a multiplicação/acumulação bate com contas feitas manualmente em Python (com inteiros, não float).
2. **`requant_unit`** + **`relu_unit`** + testbench — valide a conta de requantização contra a função Python `requantize_int32_to_int8` (já implementada), usando os `*_multiplier.txt` já exportados como entrada.
3. **`weight_rom`** genérica + testbench — confirme que os `.mem` carregam exatamente os valores que o script Python gerou.
4. **`line_buffer`** + **`window_gen`** + testbench com uma imagem sintética pequena (ex: 6x6) — valide a sequência de janelas geradas.
5. **`comparator_max2`** + árvore de comparadores para pooling + testbench.
6. **`conv_layer`** (primeira instância, 1→32 canais, 5x5) + testbench usando uma imagem real do MNIST. *Aqui você vai precisar exportar do Python não só os pesos, mas também a saída intermediária da Conv1 (antes de treinar de novo, rode `model.predict` parando na primeira camada) pra ter um valor de referência.*
7. **`maxpool_layer`** (primeira instância) + testbench, mesma lógica de comparação com saída intermediária do Python.
8. **`conv_layer`** (segunda instância, 32→64 canais) + testbench.
9. **`maxpool_layer`** (segunda instância) + testbench.
10. **`dense_layer`** (1024→10) + testbench.
11. **`argmax_unit`** + testbench.
12. **`control_fsm`** + **`cnn_top`** — integração final, testbench com uma imagem completa do MNIST, comparando a classe prevista pelo hardware com a predição do modelo Python original.

## 5. Recomendações extras

- Adicione ao seu script Python de exportação uma etapa que salva as **ativações intermediárias** (saída de cada camada para uma imagem de teste fixa), não só os pesos — sem isso você não tem como validar `conv_layer`, `maxpool_layer` e `dense_layer` isoladamente.
- Padronize desde já um protocolo simples de handshake entre camadas (ex: sinais `start`/`done` ou `valid`/`ready`) — isso evita retrabalho quando for ligar tudo no `cnn_top`.
- Antes de decidir paralelismo (item 4 da seção 2), veja quantos DSPs/LUTs sua FPGA alvo tem — isso pode forçar a arquitetura sequencial em vez da paralela.
