# Atividade de Inteligência Artificial - Implementações de Algoritmos

Este repositório contém implementações em Python de dois algoritmos fundamentais de Inteligência Artificial, desenvolvidos como parte de uma atividade para a disciplina de IA. Os algoritmos implementados são:

1.  **Árvore de Decisão:** Um algoritmo de aprendizado supervisionado utilizado para tarefas de classificação e regressão.
2.  **Perceptron:** Um algoritmo fundamental para classificação linear, sendo um dos blocos de construção de redes neurais mais complexas.

## Estrutura do Repositório

O repositório está organizado da seguinte forma:

```
├── README.md           # Este arquivo, contendo informações sobre o repositório
└── scripts/
├── arvore_decisao.py # Script Python contendo a implementação da Árvore de Decisão
└── perceptron.py     # Script Python contendo a implementação do Perceptron
```

## Descrição dos Scripts

### `arvore_decisao.py`

Este script implementa uma Árvore de Decisão para tarefas de classificação. As principais características da implementação incluem:

* **Cálculo de Entropia:** Utilizado para medir a impureza de um conjunto de dados.
* **Divisão de Dados:** Função para particionar o conjunto de dados com base em um atributo e um valor.
* **Ganho de Informação:** Métrica para determinar o melhor atributo para realizar a divisão.
* **Construção Recursiva da Árvore:** A árvore é construída de forma recursiva, dividindo os dados até atingir um critério de parada (pureza dos nós ou profundidade máxima).
* **Representação da Árvore:** Utiliza uma classe `ArvoreDeDecisao` com uma classe interna `_NoDecisao` para representar os nós da árvore.
* **Funções para Treinamento, Impressão e Classificação:** Métodos para treinar a árvore a partir de dados, exibir a estrutura da árvore e classificar novas instâncias.

### `perceptron.py`

Este script implementa um Perceptron, um algoritmo de aprendizado linear para classificação binária. As principais características da implementação incluem:

* **Função de Ativação Degrau:** Utilizada para produzir uma saída binária.
* **Treinamento Iterativo:** Os pesos do Perceptron são ajustados iterativamente com base no erro entre a saída prevista e a saída esperada.
* **Taxa de Aprendizagem e Limiar:** Parâmetros que controlam o processo de aprendizado e a decisão de classificação.
* **Representação do Perceptron:** Utiliza uma classe `Perceptron` para encapsular os pesos, o limiar e os métodos de treinamento e previsão.
* **Funções para Treinamento e Previsão:** Métodos para treinar o Perceptron com dados rotulados e para prever a classe de novas instâncias.

## Como Executar os Scripts

Para executar os scripts, você precisará ter o Python 3 instalado em seu sistema, juntamente com as bibliotecas `numpy` e `pandas` (para a Árvore de Decisão).

1.  **Clone o repositório** (se ainda não o fez).
2.  **Navegue até o diretório `scripts`:**
    ```bash
    cd scripts
    ```
3.  **Execute os scripts com o interpretador Python:**
    ```bash
    python arvore_decisao.py
    python perceptron.py
    ```

    Os scripts irão executar as implementações dos algoritmos com dados de exemplo e exibir os resultados no seu terminal.

---

**Autor:** Anthony Ricardo Rodrigues Rezende 

**Data:** 4 de Maio de 2025 \newline

**Disciplina:** Inteligência Artificial

**Instituição:** Universidade Federal de Mato Grosso