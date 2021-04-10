import numpy as np
#matriz de entradas
x = np.matrix("1 1 1; 1 -1 1; -1 -1 1; -1 1 1")
#matriz de pesos da camada oculta (primeira camada)
w = np.matrix("-1 1.5; 2 1; 1 -1")
#vetor de pesos da camada de saída (segunda camada)
wZ = np.array([-2, -1, 1])
#constante lambda (inclinação da função sigmoide) e constante alfa (taxa de treinamento)
lamb = alfa = 0.5
#constante do limiar de erro
e = 0.4
#vetor de erros inicializados com "inf" (representação de infinito)
eY = np.full(np.shape(x)[0], float("inf"))
#quantidade de neurônios da camada oculta
neuronios = 2
#vetor de saídas desejadas
d = np.array([0.5, -0.5, -0.5, -0.5])
#matriz tridimensional (duas camadas de matrizes bidimensionais) baseada na matriz de pesos da camada oculta
wAuxiliar = np.zeros((2, np.shape(x)[0], np.shape(x)[1])) 
#cálculo da matriz tridimensional de pesos
for i in range(neuronios):  
    wAuxiliar[i, :, :] = np.tile(w[:, i].transpose(), (np.shape(x)[0], 1))
w = wAuxiliar
#tranformação do vetor de pesos da camada de saída em uma matriz
wZ = np.tile(wZ, (np.shape(x)[0], 1))
#matriz de saídas intermediárias da camada oculta inicializadas com zero
s = np.zeros((np.shape(x)[0], neuronios))
#matriz de saídas da camada oculta inicializadas com zero
z = np.zeros((np.shape(x)[0], neuronios+1))
#vetor de saídas intermediárias da camada de saída inicializadas com zero
t = np.zeros((np.shape(x)[0], 1))
#vetor de saídas da camada de saída inicializadas com zero
y = np.zeros(np.shape(x)[0])
#quantidade iterações do backpropagation determinada pela entrada do usuário
iteracoes = int(input("Digite o número de iterações do backpropagation (ou 0 para considerar apenas o limiar de erro): "))
iteracao = 0
#repetição do backpropagation
while((iteracoes > 0 and (iteracao < iteracoes)) or (iteracoes <= 0 and (np.max(eY) >= e))):
    #cálculos do backpropagation
    for i in range(np.shape(x)[0]):
        #cálculo das saídas da camada oculta
        for n in range(neuronios):
            #calculo das saídas intermediárias da camada oculta
            for j in range(np.shape(x)[1]):  
                s[i, n] = s[i, n] + x[i, j] * w[n, i, j]  
            #cálculo das saídas da camada oculta
            z[i, n] = (1 - np.power(np.exp(1), (-lamb * s[i, n]))) / (1 + np.power(np.exp(1), (-lamb * s[i, n])))
        z[i, n+1] = 1
        #cálculo das saídas intermediárias da camada de saída
        for j in range(np.shape(wZ)[1]):
            t[i] = t[i] + z[i, j] * wZ[i, j]
        #cálculo das saídas da camada de saída 
        y[i] = (1 - np.power(np.exp(1), (-lamb * t[i]))) / (1 + np.power(np.exp(1), (-lamb * t[i])))
    #impressão das saídas da rede
    print(f"{iteracao+1}° Iteração do Backpropagation:", end="\n\n")
    print("Vetor de saídas da rede para cada entrada:")
    print(y, end="\n\n")
    #cálculo do erro (diferença entre a saída da rede e a saída desejada)
    eY = d - y
    #treinamento do backpropagation (atualização da matriz de pesos)
    if(np.max(eY) >= e):
        #cálculo do teta (derivada da função sigmoide) da camada de saída
        tetaY = 0.5 * lamb * (1 - np.power(y, 2))
        #cálculo do delta (erro do neurônio) da camada de saída
        deltaY = tetaY * eY
        #cálculo do sigma (variação dos pesos) da camada de saída 
        sigmaY = 2 * alfa * deltaY
        #cálculo do erro da camada oculta
        eZ = (wZ.transpose() * deltaY).transpose()
        #cálculo do teta da camada oculta
        tetaZ = 0.5 * lamb * (1 - np.power(z, 2))
        #cálculo do delta da camada oculta
        deltaZ = np.multiply(tetaZ, eZ)
        #cálculo do sigma da camada oculta
        sigmaZ = 2 * alfa * deltaZ
        #cálculo da nova matriz de pesos da camada de saída
        wZ = wZ + (sigmaY * z.transpose()).transpose()
        #cálculo da nova matriz tridimensional de pesos da camada oculta
        for i in range(np.shape(w)[0]):
            w[i, :, :] = w[i, :, :] + np.multiply(sigmaZ, x)
        print("Resultado do Treinamento:", end="\n\n")
        print("Matriz de pesos da camada de saída para cada entrada:")
        print(wZ, end="\n\n")
        for i in range(neuronios):
            print(f"Matriz de pesos da camada oculta para cada entrada ({i+1}° neurônio):")    
            print(w[i, :, :])
    print()
    iteracao += 1