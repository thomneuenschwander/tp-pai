# Trabalho Prático de Processamento e Análise de Imagens

<!-- TODO: falar sobre o projeto aqui -->

## Execução do Projeto

```bash
git clone https://github.com/thomneuenschwander/tp-pai.git # baixar o repositório
cd tp-pai

python -m venv venv # criar virtual environment

pip install -r requirements.tx # instalar dependencies

python .\spec.py 802717 805496 814939 814143 # visualizar o escopo do projeto com base nos números de matrícula dos componentes do grupo

python main.py # executar projeto
```

## Dataset OASIS Longitudinal Demographic

| Variável       | Descrição               | Tipo                    |
|----------------|-----------------------------------------------------------------------------------------------|------------------------------------|
| **Group**      | Classe do paciente: Demented (Demente), Nondemented (Não demente), Converted (Inicialmente não demente, mas tornou-se demente durante o estudo). | Categórico (Demented, Nondemented, Converted) |
| **Visit**      | Número da visita do paciente.                                                                 | Numérico Discreto                |
| **MR Delay**   | Dias entre a primeira consulta e o exame de imagem.                                           | Numérico Discreto       |
| **Age**        | Idade do paciente no momento da aquisição da imagem (em anos).                                | Numérico Discreto          |
| **Sex**        | Sexo do paciente.                                                                             | Categórico Nominal (M/F)                  |
| **Educ**       | Anos de educação formal do paciente.                                                          | Numérico Discreto          |
| **Hand**       | Mão dominante do paciente.                                                                    | Categórico (L/R)                  |
| **CDR**        | Classificação Clínica de Demência: 0 (sem demência), 0.5 (muito leve), 1 (leve), 2 (moderada). | Categórico (0, 0.5, 1, 2)         |
| **NSE**        | Status socioeconômico, avaliado pelo Índice de Posição Social de Hollingshead (1 = mais alto, 5 = mais baixo). | Categórico (1 a 5)                |
| **MMSE**       | Pontuação do Mini-Exame do Estado Mental (0 = pior, 30 = melhor).                             | Numérico Discreto       |
| **eTIV**       | Volume intracraniano total estimado (em cm³).                                                 | Numérico contínuo     |
| **nWBV**       | Volume cerebral total normalizado, expresso como percentual de voxels rotulados como substância cinzenta ou branca. | Numérico contínuo         |


### 1. Análise Exploratória dos Dados

O **EDA** *(Exploratory Data Analysis)* é o processo/etapa de investigar datasets para resumir seus dados com estatísticas descritivas e vizualizações. Dessa forma, é possível entender melhor os padrões nos dados (distribuições), identificar anomalias e descobirr relações entre as variáveis.

### 2. Preparamento e Particionamento do Conjunto de Dados

<!-- TODO: falar do particionamento 4:1 (extratificação), duas classes e da atomicidade (é logitucional) que evita data leakage. -->


## Colaboradores
| <img src="https://github.com/thomneuenschwander.png" width="100" height="100" alt="Thomas Neuenschwander"/> | <img src="https://github.com/henriquerlara.png" width="100" height="100" alt="Henrique Lara"/> | <img src="https://github.com/EduardoAVS.png" width="100" height="100" alt="Eduardo Araújo"/> | <img src="https://github.com/LuigiLouback.png" width="100" height="100" alt="Luigi Louback"/> |
|:---:|:---:|:---:|:---:|
| [Thomas <br> Neuenschwander](https://github.com/thomneuenschwander) | [Henrique <br> Lara](https://github.com/henriquerlara) | [Eduardo <br> Araújo](https://github.com/EduardoAVS) | [Luigi <br> Louback](https://github.com/LuigiLouback) |

## Fontes
- [OASIS-2 - Superset Dataset Page](https://sites.wustl.edu/oasisbrains/home/oasis-2)
- [Download MRIcron - Vizualizador de Imagens Médicas](https://people.cas.sc.edu/rorden/mricron/install.html)
- [Open access series of imaging studies: longitudinal MRI data in nondemented and demented older adults - Ref Article](https://pubmed.ncbi.nlm.nih.gov/19929323)
- [Medium - The Box Plot: A Simple but Informative Visualization](https://medium.com/analytics-vidhya/the-box-plot-a-simple-but-informative-visualization-cacc20d9ff25)