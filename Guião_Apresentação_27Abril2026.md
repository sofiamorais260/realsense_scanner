# Guião — Update 27/04/2026
> Um slide por secção. Lê os bullets e fala com as tuas próprias palavras.

---

## SLIDE 1 — Título
- Projecto de tese: plataforma 3D automática para caracterização óptica de tecido canceroso, sem marcadores químicos
- Trabalho na Champalimaud, com Dr. João Lagarto
- Hoje: update completo do pipeline, da A ao Z

---

## SLIDE 2 — Pipeline Completo
- Dois blocos: **preparação** (GRBL → Calibração → ROI → Filtros) e **scan** (Calib. Máquina → Raster → Integração óptica)
- O loop entre Filtros e ROI Analysis é propositado — sistema iterativo
- Vou percorrer cada passo

---

## SLIDE 3–4 — GRBL Controls
- GRBL = firmware de controlo de movimento (Arduino + G-code), igual ao de impressoras 3D
- **Home Machine** → robot vai para os limit switches (sabe onde está)
- **FOV Home** = posição onde a câmara vê a bandeja completa — guardada em disco, recuperável sempre
- Jog Controller: movimento manual passo a passo (1 mm/passo)

---

## SLIDE 5 — Joystick
- Joystick físico ligado a Arduino Uno separado (COM4)
- Porquê? Durante calibração, mover a sonda com precisão sub-mm com o rato é impossível
- Joystick = movimento contínuo e intuitivo
- Monitor mostra valores analógicos brutos (A0–A5) → convertidos em comandos jog GRBL

---

## SLIDE 6–7 — Calibração XY
- **Problema:** câmara vê pixéis, nós precisamos de milímetros reais
- **Solução:** tabuleiro ChArUco (6×8 quadrados de 14.5 mm)
- ChArUco = xadrez + marcadores ArUco com IDs únicos → detecta cantos mesmo com oclusão parcial

**O que é a homografia — e o que é nosso?**
- A homografia é uma matriz 3×3 que transforma coordenadas de pixel em coordenadas reais em mm no plano do tabuleiro
- A função `cv2.findHomography` (OpenCV) calcula essa matriz — é da biblioteca
- **O que é nosso:** todo o pipeline à volta — a lógica de detecção e validação dos cantos, a correcção de distorção da lente antes de computar a homografia (`_undistort_points`), o ajuste do plano 3D por SVD, a detecção dos planaltos da escada, a regressão linear da calibração Z, a avaliação dos resíduos, e todo o sistema de persistência com histórico — isso é tudo código escrito por mim

**O que é o RMSE — e como ler os valores**
- RMSE = Root Mean Square Error = raiz do erro médio quadrático
- Intuitivamente: é o erro típico esperado — se o RMSE for 0.08 mm, significa que em média os pontos reprojectados erram 0.08 mm relativamente às posições reais medidas
- Preferimos o RMSE ao erro máximo porque é mais robusto a outliers; o erro máximo diz-nos o pior caso
- **RMSE XY = 0.0835 mm** → os 35 cantos ChArUco reprojectados pela homografia ficam a menos de 0.1 mm das posições reais → excelente

- **Resultados desta calibração:** 12/12 aquisições OK, escala **0.3842 mm/px**, RMSE **0.0835 mm**
- **Porquê guardar?** Só muda se movermos a câmara — carrega calibração anterior enquanto nada mudar

---

## SLIDE 8 — Calibração Z
- **Problema:** profundidade bruta da câmara ≠ altura real em mm
- **Solução:** pirâmide em escada com degraus de alturas conhecidas (9.2, 19.2, 29.2, 39.3 mm)
- Algoritmo detecta os planaltos automaticamente (histograma + clustering, código nosso) → regressão linear `np.polyfit` (NumPy) com os pares medido/real
- **Equação:** y = 1.0032x − 0.1561 mm
  - 1.0032 = escala quase 1:1 (câmara muito bem calibrada)
  - −0.1561 = bias sistemático de menos de 0.2 mm → corrigido automaticamente em todas as medições seguintes
- **RMSE do ajuste: 0.123 mm** → os 4 degraus caem quase em cima da recta → boa linearidade
- **Porquê guardar?** Mesma razão: só muda se mudar a distância câmara–bandeja

---

## SLIDE 9–10 — ROI Selection
- **Manual:** utilizador desenha rectângulo na imagem a cores em tempo real
- **Automático:** algoritmo detecta objecto quente (tons rosa/vermelho) sobre fundo azul, via HSV
  - Fundo azul = contraste intencional com tecido biológico
  - Aplica morfologia (blur + open/close) para remover ruído
  - Pede confirmação antes de aceitar

---

## SLIDE 11 — ROI Tracking
- **Locked ROI:** rectângulo fixo — usa-se com amostra imóvel
- **Unlocked ROI:** tracking frame a frame, actualiza posição automaticamente
  - Só actualiza se confiança > 0.70, senão mantém última posição válida

---

## SLIDE 12–13 — Filter Selection
- Câmara usa estereoscopia IR → gera ruído (speckle, pixéis inválidos, flutuações)
- 4 filtros principais, todos configuráveis:

| Filtro | Parâmetro chave | Porquê este valor |
|---|---|---|
| **Espacial** | alpha=55, delta=20 | Equilibrio suavização vs. preservação de arestas |
| **Temporal** | alpha=40, delta=20, persist=8 | Mais peso ao histórico; 8 frames de persistência para oclusões |
| **Decimação** | magnitude=2 | Metade da resolução, 4× mais rápido |
| **Threshold** | min=130, max=150 mm | Distância real câmara–bandeja na nossa montagem |
| **Hole Filling** | modo=1 (farthest) | Em tecido, vizinho mais distante é estimativa mais conservadora |

---

## SLIDE 14 — Comparação Visual de Filtros
- Mesmo objecto, 6 combinações de filtros → diferença gritante
- **Sem filtros:** muito ruído, bordas irregulares
- **Temporal:** zonas planas melhoram, bordas ainda ruidosas
- **Decimação:** suave mas perde definição
- **Todos:** muito suavizado, pode perder detalhes reais
- **Temporal + Threshold + Spatial → melhor equilíbrio** (bordas limpas, boa cobertura, pouco ruído)

---

## SLIDE 15–16 — Depth Profile
- Captura linha de profundidade horizontal pelo centro da ROI em tempo real
- Útil para diagnóstico rápido antes do scan completo:
  - Muito ruído → ajustar filtros
  - Plano quando devia ter estrutura → threshold mal configurado
- Pirâmide: degraus claramente visíveis como onda em escada

---

## SLIDE 17 — Preset + Filter Suggestion (automático)
- **Problema:** 3 presets × 6 combinações de filtros = 18 testes → impossível fazer à mão
- Sistema testa automaticamente as 18 combinações (5 frames cada, ~20–40 seg total)
- Métricas de scoring: cobertura, ruído temporal, nitidez de arestas, erro de pico
- **Resultados (alvo 39.7 mm):**
  1. Default | Temporal → score **81.5**, erro pico 0.252 mm ✓
  2. Default | Spatial + Temporal + Decimation → score 74.4
  3. High Density | Spatial + Temporal + Hole filling → score 72.5
- Aplica o melhor com um clique; pode fazer Repeatability Check (5×)

---

## SLIDE 18 — Topography Map
- Para cada pixel da ROI: câmara dá coordenadas 3D → calibração converte para altura acima do plano
- Mapa 2D colorido: azul=baixo, vermelho=alto
- **Resultados com a pirâmide:** ROI 63.4×62.7 mm, altura medida **39.38 mm** (real: 39.7 mm) → **erro de 0.32 mm**
- Valida a qualidade da calibração Z

---

## SLIDE 19–20 — Machine Calibration
- **Problema:** a câmara sabe onde as coisas estão em mm no tabuleiro; o robot não sabe o que é o tabuleiro — os dois sistemas têm origens e orientações completamente diferentes
- **Solução:** registar os dois sistemas usando pontos conhecidos pelos dois

- **Processo:**
  1. Colocar o tabuleiro ChArUco na bandeja → câmara detecta os cantos e sabe as posições em mm
  2. Mover a sonda com o joystick até estar centrada visualmente sobre cada canto → robot regista a sua posição nesse momento
  3. Repetir para **26 cantos** → tenho 26 pares "posição no tabuleiro ↔ posição do robot"
  4. O software encontra a melhor transformação que converte qualquer ponto do tabuleiro em posição do robot — usando mínimos quadrados sobre os 26 pares

- **RMSE 0.476 mm** → o erro típico é de meio milímetro — aceitável porque o scan tem espaçamento de 1 mm entre linhas
- **Máximo 0.888 mm** → o pior dos 26 pontos ainda fica abaixo de 1 mm
- **Rotação −90.356°** → a câmara está fisicamente rodada 90° relativamente ao robot; a transformação absorve isso automaticamente

---

## SLIDE 21–22 — Raster Scan com Z Fixo
- Com calibração feita: software projecta ROI → coordenadas tabuleiro → coordenadas máquina
- Trajectória em **serpentina** (zigzag) → minimiza deslocamento total
- Log de cada ponto: timestamp Unix, índice de amostra, linha, X/Y/Z da máquina
- Z **fixo** = working offset constante acima do plano
- Na imagem: tecido animal real, sonda de fibra visível, ponto verde = contacto da fibra

---

## SLIDE 23–24 — Integração com Sistema Óptico
- **Primeira vez que os dois sistemas correm juntos**
- pyProbe = software de controlo do espectrofluorómetro (Champalimaud)
- Cada ponto de scan: **15.000 medições de fotões**, resolução 200 ps, integração 15 s → histograma de decaimento FLIM (TCSPC)
- Scan completo: **32 linhas** sobre tecido animal
- Outputs gerados: ch1.h5, ch2.h5 (dados dos 2 detectores), log, vídeo
- **"Need to analyse results"** → experiência correu, ficheiros gerados, análise a seguir

---

## SLIDE 25 — Pipeline (revisão)
- Pipeline completo de ponta a ponta: funcionou, testado em condições reais
- [transição rápida para o slide final]

---

## SLIDE 26 — Problems & Next Steps

| # | Problema | Próximo passo |
|---|---|---|
| A | Scan e fluorescência desconectados | Merge por timestamp Unix |
| B | FLIM não sincronizado com posição | Sincronização robusta por timestamp |
| C | Z fixo (não segue topografia) | Z adaptativo baseado no mapa de topografia |
| D | Amostras altas → colisão do carro | Collision-aware height handling |
| E | Fundo azul fluoresce | Substituir por fundo preto, adaptar detecção ROI |
| F | Vídeo não gerado de forma fiável | Reimplementar à base do pyProbe |

---

## SLIDE 27 — Final
- Pipeline completo e validado
- Primeira integração óptica feita com sucesso
- Próximos meses: fechar análise + espécimens cirúrgicos reais na histopatologia
- **"Estou disponível para perguntas"**
