import numpy as np
import pandas as pd
from collections import Counter
import math
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

class ArvoreDeDecisao:
    def __init__(self, profundidade_maxima=5):
        self.profundidade_maxima = profundidade_maxima
        self.raiz = None
        self.nomes_atributos = None

    def _calcular_entropia(self, conjunto_dados):
        rotulos = conjunto_dados[:, -1]
        contagem_rotulos = Counter(rotulos)
        total_amostras = len(rotulos)
        entropia = 0.0
        for contagem in contagem_rotulos.values():
            probabilidade = contagem / total_amostras
            entropia -= probabilidade * math.log2(probabilidade)
        return entropia

    def _dividir_dados(self, conjunto_dados, indice_atributo, valor_divisao):
        esquerda = conjunto_dados[conjunto_dados[:, indice_atributo] <= valor_divisao]
        direita = conjunto_dados[conjunto_dados[:, indice_atributo] > valor_divisao]
        return esquerda, direita

    def _calcular_ganho_informacao(self, conjunto_dados, indice_atributo, valor_divisao):
        total_amostras = len(conjunto_dados)
        esquerda, direita = self._dividir_dados(conjunto_dados, indice_atributo, valor_divisao)
        if not esquerda.size or not direita.size:
            return 0
        ganho = self._calcular_entropia(conjunto_dados)
        ganho -= (len(esquerda) / total_amostras) * self._calcular_entropia(esquerda)
        ganho -= (len(direita) / total_amostras) * self._calcular_entropia(direita)
        return ganho

    def _encontrar_melhor_divisao(self, conjunto_dados):
        melhor_ganho = 0
        melhor_indice_atributo = -1
        melhor_valor_divisao = None
        num_atributos = conjunto_dados.shape[1] - 1

        for indice_atributo in range(num_atributos):
            valores_unicos = np.unique(conjunto_dados[:, indice_atributo])
            valores_ordenados = np.sort(valores_unicos)
            pontos_divisao = (valores_ordenados[:-1] + valores_ordenados[1:]) / 2 if len(valores_ordenados) > 1 else valores_ordenados

            for valor in pontos_divisao:
                ganho = self._calcular_ganho_informacao(conjunto_dados, indice_atributo, valor)
                if ganho > melhor_ganho:
                    melhor_ganho = ganho
                    melhor_indice_atributo = indice_atributo
                    melhor_valor_divisao = valor
        return melhor_indice_atributo, melhor_valor_divisao

    class _NoDecisao:
        def __init__(self, indice_atributo=None, valor_divisao=None, esquerda=None, direita=None, valor_folha=None):
            self.indice_atributo = indice_atributo
            self.valor_divisao = valor_divisao
            self.esquerda = esquerda
            self.direita = direita
            self.valor_folha = valor_folha

    def _construir_arvore_recursivamente(self, conjunto_dados, profundidade):
        rotulos = conjunto_dados[:, -1]
        if len(np.unique(rotulos)) == 1 or len(conjunto_dados) == 0 or profundidade >= self.profundidade_maxima:
            return self._NoDecisao(valor_folha=Counter(rotulos).most_common(1)[0][0] if rotulos.size else None)

        melhor_indice_atributo, melhor_valor_divisao = self._encontrar_melhor_divisao(conjunto_dados)

        if melhor_indice_atributo == -1:
            return self._NoDecisao(valor_folha=Counter(rotulos).most_common(1)[0][0])

        esquerda_dados, direita_dados = self._dividir_dados(conjunto_dados, melhor_indice_atributo, melhor_valor_divisao)

        esquerda_filho = self._construir_arvore_recursivamente(esquerda_dados, profundidade + 1)
        direita_filho = self._construir_arvore_recursivamente(direita_dados, profundidade + 1)

        return self._NoDecisao(indice_atributo=melhor_indice_atributo, valor_divisao=melhor_valor_divisao, esquerda=esquerda_filho, direita=direita_filho)

    def treinar(self, dados_treino, nomes_atributos):
        self.nomes_atributos = nomes_atributos
        self.raiz = self._construir_arvore_recursivamente(dados_treino, 0)

    def _imprimir_arvore_recursivamente(self, no, profundidade, ramo=""):
        indentacao = "  " * profundidade
        if no.valor_folha is not None:
            print(f"{indentacao}{ramo}Classe: {no.valor_folha}")
        else:
            print(f"{indentacao}{ramo}Atributo: {self.nomes_atributos[no.indice_atributo]} <= {no.valor_divisao:.2f}")
            self._imprimir_arvore_recursivamente(no.esquerda, profundidade + 1, "Sim -> ")
            self._imprimir_arvore_recursivamente(no.direita, profundidade + 1, "Não -> ")

    def imprimir_arvore(self):
        print("\nÁrvore de Decisão Resultante:")
        self._imprimir_arvore_recursivamente(self.raiz, 0)

    def _classificar_instancia_recursivamente(self, instancia, no):
        if no.valor_folha is not None:
            return no.valor_folha
        else:
            nome_atributo = self.nomes_atributos[no.indice_atributo]
            valor_atributo = instancia[self.nomes_atributos.index(nome_atributo)]
            if valor_atributo <= no.valor_divisao:
                return self._classificar_instancia_recursivamente(instancia, no.esquerda)
            else:
                return self._classificar_instancia_recursivamente(instancia, no.direita)

    def classificar(self, instancia):
        return self._classificar_instancia_recursivamente(instancia, self.raiz)

print("\n--- Construção da Árvore de Decisão ---")

# Conjunto de treinamento para a árvore de decisão
dados_treinamento = {
    'Caracteristica_1': [200, 125, 87, 23, 348, 85, 75, 5, 127, 210, 100, 57],
    'Caracteristica_2': [4, 7, 1, 9, 3, 7, 2, 8, 1, 3, 6, 5],
    'Caracteristica_3': [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    'Rotulo': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  # Classes definidas pelo meu RGA - 202211310048
}

dataframe_arvore = pd.DataFrame(dados_treinamento)
dados = dataframe_arvore.to_numpy()
nomes_caracteristicas = ['Caracteristica_1', 'Caracteristica_2', 'Caracteristica_3']

# Criar e treinar a árvore de decisão
arvore = ArvoreDeDecisao()
arvore.treinar(dados, nomes_caracteristicas)
arvore.imprimir_arvore()

# Classificar todas as instâncias do conjunto de treinamento
print("\nClassificações no Conjunto de Treinamento:")
for i in range(len(dataframe_arvore)):
    amostra = dataframe_arvore.iloc[i][['Caracteristica_1', 'Caracteristica_2', 'Caracteristica_3']].to_list()
    classe_prevista = arvore.classificar(amostra)
    classe_real = dataframe_arvore.iloc[i]['Rotulo']
    print(f"Instância {i+1}: Características={amostra}, Previsto={classe_prevista}, Real={classe_real}")