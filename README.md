# Projeto-de-Aplicacao
1. Motivação e Objetivo
Contexto e Problema de Saúde:
As infecções do trato urinário (ITUs) representam uma das principais causas de prescrição de antibióticos na atenção primária e hospitalar. O crescimento contínuo de organismos resistentes a antimicrobianos (RAM/AMR) é agravado pelo uso frequente e excessivo de antibioticoterapia de amplo espectro.
Relevância Clínica:
Os exames microbiológicos com Testes de Suscetibilidade a Antimicrobianos (TSA) levam tipicamente até 72 horas para disponibilizar o perfil de resistência definitivo. Durante essa janela crítica de espera, o médico necessita prescrever um tratamento empírico. A decisão clínica exige equilibrar o uso preferencial de antibióticos de espectro estreito (alinhado às diretrizes de Antimicrobial Stewardship) contra o risco de prescrição inadequada (prescrever um fármaco ao qual o patógeno é resistente, resultando em falha terapêutica).
Objetivo Geral e Uso Pretendido:
Desenvolver e avaliar modelos de Aprendizado de Máquina integrados a um Sistema de Suporte à Decisão Clínica (CDSS) para prever a suscetibilidade a antibióticos específicos (Nitrofurantoína - NIT, Trimetoprima-Sulfametoxazol - SXT, Ciprofloxacina - CIP e Levofloxacina - LVX) no momento exato da coleta de urina. O sistema visa recomendar o antibiótico de menor espectro eficaz, beneficiando prescritores e pacientes com tratamentos otimizados e reduzindo a pressão seletiva para superbactérias.
2. Descrição do Dataset
2.1. Descrição dos Dados
O projeto utiliza o conjunto de dados AMR-UTI (Antimicrobial Resistance in Urinary Tract Infections), derivado de registros eletrônicos de saúde do Massachusetts General Hospital e do Brigham & Women's Hospital (Boston, MA, EUA) entre os anos de 2007 e 2016. A coorte compreende mais de 80.000 pacientes únicos com episódios de ITU.

butoDemografia	demographics_age, raça, sexo e status de veterano.
Histórico Microbiológico	micro_prev_resistance_* (infecções anteriores e perfis prévios de resistência microbiológica).
Histórico de Medicações	medication_* (prescrições recentes de antibióticos e outros fármacos).
Comorbidades	comorbidity_* (índices de comorbidade baseados na classificação de Elixhauser).
Contexto Hospitalar	hosp_ward_* (tipo de setor/departamento hospitalar: IP, OP, ER, ICU).
Pressão de Colonização	selected_micro_colonization_pressure_* (taxa local e recente de resistência no ambiente hospitalar).
Rótulo / Target Binário (y)	NIT, SXT, CIP, LVX (0 = Suscetível, 1 = Resistente/Intermediário). Unidade de observação: Amostra/espécime de urina.
2.2. Etapas de Preparação dos Dados e Pré-processamento
•	Preenchimento de Valores Ausentes (Missing Values): Na ausência de histórico prévio registrado de medicações ou infecções, assumiu-se 0 implícito (ausência de exposição/resistência antecedente). Para variáveis demográficas com dados ausentes (como raça, ~3%), aplicou-se imputação categórica de valor padrão.
•	Prevenção de Vazamento de Dados (Data Leakage): Filtragem rigorosa para assegurar que apenas dados anteriores ao exame fossem incluídos. Registros microbiológicos obtidos a < 7 dias ou medicações administradas a < 2 dias do exame foram desconsiderados na construção dos atributos do predictor.
•	Agrupamento e Resolução de Amostras Múltiplas: Fusão de amostras do mesmo paciente coletadas em intervalos inferiores a 14 dias para evitar duplicidade de episódios infecciosos e viés de dependência temporal.
•	Normalização e Codificação: Codificação one-hot de variáveis categóricas (setores hospitalares) e normalização min-max / z-score para atributos numéricos contínuos como idade e pressão de colonização local.
2.3. Link para a Versão Disponibilizada do Dataset
• Nome do Dataset: AMR-UTI: Antimicrobial Resistance in Urinary Tract Infections (v1.0.0)
• Repositório / Fonte: PhysioNet
• Link / DOI: https://doi.org/10.13026/se6w-f455
• Requisitos de Acesso: Acesso restrito/credenciado (necessita assinatura de Data Use Agreement - DUA e conclusão do treinamento de ética em pesquisa CITI Program).
3. Modelo de Aprendizado de Máquina
3.1. Formalização Matemática
O problema de previsão de resistência a antibióticos é formulado como uma tarefa de Classificação Binária Multirrótulo (Multi-label Binary Classification) e de Aprendizado de Políticas / Suporte à Decisão Causal (Policy Learning).
Para cada amostra i de urina associada a um vetor de características X_i ∈ ℝ^d (demografia, histórico prévio, comorbidades e contexto hospitalar), estimam-se as probabilidades condicionais de resistência para cada antibiótico k ∈ {NIT, SXT, CIP, LVX}:

    P(Y_i,k = 1 | X_i) = f_k(X_i; θ_k)

Onde Y_i,k ∈ {0, 1} indica a resistência ao antibiótico k (1 = Resistente/Intermediário, 0 = Suscetível), e f_k representa a função treinada com parâmetros θ_k (modelada via Regressão Logística regularizada, Gradient Boosted Trees ou Redes Neurais).

Na camada de Aprendizado de Políticas (Policy Learning), define-se uma função de decisão π(X_i) que seleciona o antibiótico a_i de menor espectro que satisfaça o critério de eficácia empírica esperada:

    π(X_i) = argmin_{a ∈ A_eficaz} Espectro(a),   sujeito a P(Y_i,a = 0 | X_i) ≥ τ

onde τ representa o limiar clínico mínimo de probabilidade de suscetibilidade aceitável.
3.2. Método de Validação
Para simular realisticamente a aplicação clínica em tempo real e evitar vazamento temporal, adotou-se o método de Validação Temporariamente Estruturada (Out-of-Time Validation). Os dados de amostragem foram divididos cronologicamente:
• Treinamento e Ajuste de Hiperparâmetros: Episódios entre 2007 e 2013 (com validação cruzada k-fold temporal internamente).
• Teste Final Prospectivo: Episódios entre 2014 e 2016 (sem sobreposição de pacientes da coorte entre treino e teste).
3.3. Medidas de Desempenho
•	AUROC (Area Under the Receiver Operating Characteristic Curve): Mede a capacidade global de discriminação do modelo entre amostras resistentes e suscetíveis para cada antibiótico.
•	AUPRC (Area Under the Precision-Recall Curve): Métrica primária para lidar com o desbalanceamento das classes de resistência entre os fármacos.
•	Acurácia da Prescrição Empírica Adequada: Proporção de tratamentos empíricos escolhidos pelo algoritmo aos quais a bactéria identificada no TSA era realmente suscetível.
•	Taxa de Redução do Uso de Amplo Espectro: Percentual de redução na prescrição de antibióticos de amplo espectro (ex.: Ciprofloxacina/Levofloxacina) em comparação com a decisão histórica dos médicos.
4. Avaliação
4.1. Amostras Usadas para Treinamento, Validação e Teste
Conjunto de Dados	Período Temporal	Tamanho da Amostra (Pacientes)
Treinamento / Validação interna	2007 – 2013	55.078 pacientes
Teste Final (Out-of-Time)	2014 – 2016	26.807 pacientes
Total da Coorte AMR-UTI	2007 – 2016	> 81.885 pacientes
4.2. Resultados das Medidas de Desempenho
A avaliação comparou a política recomendada pelo algoritmo de Machine Learning contra o baseline histórico das prescrições médicas registradas no conjunto de teste (2014-2016):
• Discriminação (AUROC/AUPRC): Os modelos obtiveram AUROCs entre 0.70 e 0.82 para a previsão de resistência individual aos quatro fármacos avaliados, com destaque para maior capacidade preditiva em histórico de resistência prévia recente.
• Adequação da Terapia Empírica: O algoritmo manteve ou elevou a taxa global de prescrição empírica adequada quando comparado às escolhas dos clínicos.
• Redução de Fármacos de Amplo Espectro: A política de suporte à decisão reduziu substancialmente o uso desnecessário de fluoroquinolonas (Ciprofloxacina e Levofloxacina), promovendo a substituição por Nitrofurantoína em episódios não complicados.
5. Conclusão
O trabalho demonstra a viabilidade e a relevância da aplicação de Machine Learning e Aprendizado de Políticas (Policy Learning) no suporte à tomada de decisão para prescrição empírica de antibióticos em ITUs. Ao integrar dados do histórico do paciente no momento da coleta do exame de urina, o modelo é capaz de anteceder a resposta microbiológica (que leva até 72 horas).
Os principais avanços incluem a capacidade de promover a stewardship de antimicrobianos ao redirecionar prescrições de amplo espectro para tratamentos de menor espectro sem comprometer a taxa de cobertura terapêutica eficaz. Trabalhos futuros envolvem a validação externa em instâncias de hospitais universitários no Brasil (como HUAP/Ebserh) e a auditoria de equidade (equity auditing) do modelo em subpopulações demográficas vulneráveis.
6. Referências
Kanjilal, S., Oberst, M., Boominathan, S., Zhou, H., Hooper, D. C., & Sontag, D. (2020). A decision algorithm to promote outpatient antimicrobial stewardship for uncomplicated urinary tract infections. Science Translational Medicine, 12(568).
Oberst, M., Boominathan, S., Zhou, H., Kanjilal, S., & Sontag, D. (2020). AMR-UTI: Antimicrobial Resistance in Urinary Tract Infections (version 1.0.0). PhysioNet. https://doi.org/10.13026/se6w-f455.
Clinical Laboratory Standards Institute (CLSI). (2017). Performance Standards for Antimicrobial Susceptibility Testing (27th ed.). CLSI supplement M100.
