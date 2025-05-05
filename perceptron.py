import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

class Perceptron:
    def __init__(self, taxa_aprendizagem=0.1, limiar=0.0, pesos_iniciais=None):
        self.taxa_aprendizagem = taxa_aprendizagem
        self.limiar = limiar
        self.pesos = pesos_iniciais

    def _funcao_ativacao(self, soma_ponderada):
        return 1 if soma_ponderada >= 0 else 0

    def treinar(self, dados_treino, rotulos_treino, numero_iteracoes=1):
        num_amostras, num_atributos = dados_treino.shape
        if self.pesos is None:
            self.pesos = np.zeros(num_atributos)

        print("Início do Treinamento do Perceptron:")
        print(f"Pesos iniciais: w={self.pesos}, Limiar={self.limiar}")

        for iteracao in range(numero_iteracoes):
            print(f"\nIteração {iteracao + 1}:")
            for indice_amostra in range(num_amostras):
                entrada = dados_treino[indice_amostra]
                rotulo_esperado = rotulos_treino[indice_amostra]

                soma_ponderada = np.dot(entrada, self.pesos) - self.limiar
                saida_prevista = self._funcao_ativacao(soma_ponderada)

                print(f"\n  Amostra {indice_amostra + 1}:")
                print(f"    Entradas: {entrada}")
                print(f"    Saída esperada: {rotulo_esperado}")
                print(f"    Soma ponderada (u): {soma_ponderada:.2f}")
                print(f"    Saída da rede (y): {saida_prevista}")

                if saida_prevista != rotulo_esperado:
                    ajuste = self.taxa_aprendizagem * entrada * (rotulo_esperado - saida_prevista)
                    self.pesos += ajuste
                    print(f"    Ajuste de pesos (Δw): {ajuste}")
                    print(f"    Novos pesos: w={self.pesos}")
                else:
                    print("    Nenhum ajuste de pesos necessário.")

        print("\nPesos Finais do Perceptron:", self.pesos)

    def prever(self, dados_teste):
        print("\nTeste do Perceptron:")
        previsoes = []
        for indice_teste, amostra in enumerate(dados_teste):
            soma_ponderada = np.dot(amostra, self.pesos) - self.limiar
            saida = self._funcao_ativacao(soma_ponderada)
            previsoes.append(saida)
            print(f"\n  Amostra de Teste {indice_teste + 1}:")
            print(f"    Entradas: {amostra}")
            print(f"    Saída da rede: {saida}")
        return previsoes

print("\n--- Implementação do Perceptron ---")

# Definições do Perceptron
taxa_aprender = 0.5
valor_limiar = 0.3
pesos_comeco = np.array([2.0, 0.0, 4.0])

# Dados de treinamento
dados_treino = np.array([
    [5, 8, 3],
    [4, 9, 2],
    [6, 2, 7],
    [2, 1, 4],
    [7, 3, 5]
], dtype=float)
rotulos_treino = np.array([1, 0, 0, 1, 1])

# Dados de teste
dados_para_teste = np.array([
    [5, 2, 8],
    [1, 4, 2]
], dtype=float)
rotulos_para_teste = np.array([1, 0])

# Criar e treinar o Perceptron
modelo_perceptron = Perceptron(taxa_aprendizagem=taxa_aprender, limiar=valor_limiar, pesos_iniciais=pesos_comeco)
modelo_perceptron.treinar(dados_treino, rotulos_treino, numero_iteracoes=2)

# Testar o Perceptron
previsoes_teste = modelo_perceptron.prever(dados_para_teste)

print("\nResultados do Teste:")
for i in range(len(dados_para_teste)):
    print(f"Amostra {i + 1}: Esperado={rotulos_para_teste[i]}, Previsto={previsoes_teste[i]}")