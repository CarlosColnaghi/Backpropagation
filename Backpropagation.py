import numpy as np

x = np.matrix("1 1 1; 1 -1 1; -1 -1 1; -1 1 1")
w = np.matrix("-1 1.5; 2 1; 1 -1")
wZ = np.array([-2, -1, 1])
lamb = alfa = 0.5
e = 0.4
eY = np.full(np.shape(x)[0], float("inf"))
neuronios = 2
d = np.array([0.5, -0.5, -0.5, -0.5])

wAuxiliar = np.zeros((2, np.shape(x)[0], np.shape(x)[1])) 
for i in range(neuronios):  
    wAuxiliar[i, :, :] = np.tile(w[:, i].transpose(), (np.shape(x)[0], 1))
w = wAuxiliar
wZ = np.tile(wZ, (np.shape(x)[0], 1))
s = np.zeros((np.shape(x)[0], neuronios))
z = np.zeros((np.shape(x)[0], neuronios+1))
t = np.zeros((np.shape(x)[0], 1))
y = np.zeros(np.shape(x)[0])

iteracoes = int(input("Digite o número de iterações da retropropagação: "))
iteracao = 0
repetir = True
while(repetir):
    for i in range(np.shape(x)[0]):
        for n in range(neuronios):
            for j in range(np.shape(x)[1]):
                s[i, n] = s[i, n] + x[i, j] * w[n, i, j]  
            z[i, n] = (1 - np.power(np.exp(1), (-lamb * s[i, n]))) / (1 + np.power(np.exp(1), (-lamb * s[i, n])))
        z[i, n+1] = 1
        for j in range(np.shape(wZ)[1]):
            t[i] = t[i] + z[i, j] * wZ[i, j]
        y[i] = (1 - np.power(np.exp(1), (-lamb * t[i]))) / (1 + np.power(np.exp(1), (-lamb * t[i])))

    eY = d - y

    tetaY = 0.5 * lamb * (1 - np.power(y, 2))
    deltaY = tetaY * eY
    sigmaY = 2 * alfa * deltaY
    ez = (wZ.transpose() * deltaY).transpose()
    #ez = (np.multiply(wZ, np.matrix(deltaY).transpose()))
    tetaZ = 0.5 * lamb * (1 - np.power(z, 2))
    deltaZ = np.multiply(tetaZ, ez)
    sigmaZ = 2 * alfa * deltaZ
    wZ = wZ + (sigmaY * z.transpose()).transpose()
    #wZ = wZ + np.multiply(np.matrix(sigmaY).transpose(), z)

    for i in range(np.shape(w)[0]):
        w[i, :, :] = w[i, :, :] + np.multiply(sigmaZ, x)
    iteracao += 1
    print(iteracao)
    if(iteracoes > 0):
        if(iteracao >= iteracoes):
            repetir = False
    else:
        if(np.max(eY) < e):
            repetir = False