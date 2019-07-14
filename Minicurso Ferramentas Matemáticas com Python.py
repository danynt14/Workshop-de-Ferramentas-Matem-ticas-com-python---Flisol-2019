#Minicurso Ferramentas Matemáticas com Python

''' 
Neste curso vamos aprender um pouco de computacao cientifica com Numpy, Scipy
e matplotlib, bibliotecas inspiradas na ferramenta Matlab.
'''
#preparando o ambiente
'''
python3.7 -m pip install jupyter
python3.7 -m pip install notebook

python3.7 -m pip install iPython
python3.7 -m jupyter --version

python3.7 -m pip install numpy
python3.7 -m pip install scipy
python3.7 -m pip install matplotlib
'''ph.floyd_warshall(csgraph=grafo, 
directed=False, return_predecessors=True)#N-N

#Numpy
'''
Poderosa biblioteca de criação arrays N-dimensionais, com funções sofisticadas
sendo base para outras bibliotecas como Opencv, Scipy, Pandas. Pode ser integrado
 com C/C++ e Fortran e 
possui módulos para execução de operações sofisticadas de algebra linear, numeros
aleatórios, transformada de Fourier, e outros.

Licença: BSD licensed

Instalação: 
pip install numpy
Python 3.7 -m pip install numpy

Site para obtenção da biblioteca:
https://pypi.org/project/numpy/

Página Oficial: https://www.numpy.org/
'''


#%%
import numpy as np

print(np.__version__)

vetor1D = np.arange(20) #(start,stop.step)
matriz2D = np.arange(20).reshape(4,5)

vetor1D_booleano = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],dtype=bool)
matriz2D_complexo = np.array([[0,1,2,3,4],[5,6,7,8,9],
[10,11,12,13,14], [15,16,17,18,19]],dtype=complex)

matriz3D = np.arange(24).reshape(2,3,4)

matriz4D = np.random.randn(4, 3, 3, 2)

#atributos
'''
matriz3D.ndim
matriz3D.shape
matriz3D.size
matriz3D.dtype
'''

#funcoes especiais

matriz5DZero = np.zeros((3,4,2,3,4))
matriz2DUm = np.ones((2,2))
matriz2DLixoMemoria = np.empty((2,3))
matriz2DRandomicos = np.random.randint(5, size=(2, 2))
matriz2DIdentidade = np.eye(8)

#Operações Matemáticas - matriz2DUm/matriz2DRandomicos


print(matriz2DUm)
print(matriz2DRandomicos)

somaMatriz =  matriz2DRandomicos + matriz2DUm
subMatriz =  matriz2DRandomicos - matriz2DUm
expMatriz =  matriz2DRandomicos ** 2
booleanaMatriz = matriz2DRandomicos > 2
eqMatriz = np.sqrt(matriz2DRandomicos) * 1j
multMatriz = matriz2DUm @ matriz2DRandomicos

testeMatriz = np.arange(12).reshape(3,4)
somaColuna = testeMatriz.sum(axis=0)
somaLinha = testeMatriz.sum(axis=1)



#ALgebra Linear
matrizTransposta = testeMatriz.T

##indexação
#arrays boleanos

#Distribuição de densidade de probabilidade normal

#%%
import numpy as np
import matplotlib.pyplot as plt

media, variancia, populacao = 2, 0.5,10000
v = np.random.normal(media,variancia,populacao)

plt.hist(v, bins=50, density=1) 
#plt.plot(v) #ruido gaussiano
            #sinal de ruido cujas as amplitudes  
            #tem comportamento gaussiano  
            # -Sinal Aleatório
(n, bins) = np.histogram(v, bins=50, normed=True)# NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)

                  
plt.show()

#Operacoes com Binarios

bitOfbitAnd = np.bitwise_and(24,56)
bitOfbitOr = np.bitwise_or(24,56)
bitOfbitXor = np.bitwise_xor(24,56)
bitOfbitNot = np.bitwise_not(2)

resulOpBit = np.binary_repr(334, width=10)




bitOfbitNotBool = np.invert(np.array([[True,False,False, True],
                                      [False,False,True, True],
                                      [True,False,False, False],
                                      [False,False,False, True]]))


#Construcao de Funcao Polinomial
#%%
import numpy as np


f2grau = np.poly1d([1, 2, 3])#coeficientes
f2OtherVariable = np.poly1d([1,2,3], variable='z')
print(p(1))
f3grau = np.poly1d([1, 2, 3], True)#raizes

f3grau = np.poly1d([1,1,1,1])

#derivar


derivadaPrimeira = np.polyder(f3grau)

derivadaSegunda = np.polyder(f3grau, 2)

derivadaTerceira = np.polyder(f3grau, 3)

#integrar

integralPrimeira = np.polyint(f3grau)

integralSegunda = np.polyint(f3grau, 2)

integralTerceira = np.polyint(f3grau, 3)


#Tratamento de sinais - convolucao
#%%
sinalConvoluido = np.convolve([1, 2, 3], [0, 1, 0.5])
plt.hist(sinalConvoluido, bins=3, density=100) #altere os bins
plt.show()

#Pesquisa

print(np.info(np.polyval))
print(np.lookfor('binary representation'))
print(np.source(np.interp))#funcao Valida apenas para objetos escritos em python


'''Scipy

É uma biblioteca de computação científica, que junto ao Numpy é capaz
de realizar operações poderosas, tanto para o processamento de dados 
quanto para a prototipagem de sistemas.'''



#%%
import scipy as sp

grafo = [[0,1,2,0],[0,0,0,1],[0,0,0,3],[0,0,0,0]] 

grafo = sp.sparse.csr_matrix(grafo)

print(grafo)

dist_matrix, predecessors= sp.sparse.csgraph.dijkstra(csgraph=grafo, 
directed=False, indices=0,return_predecessors=True)#MENOR CAMINHO 1-N

dist_matrix, predecessors= sp.sparse.csgraph.floyd_warshall(csgraph=grafo, 
directed=False, return_predecessors=True)#N-N

dist_matrix, predecessors= sp.sparse.csgraph.bellman_ford(csgraph=grafo, 
directed=False, indices=0,return_predecessors=True)#PESONEGATIVO 

dist_matrix, predecessors= sp.sparse.csgraph.johnson(csgraph=grafo, 
directed=False, indices=0,return_predecessors=True)




#%%

#https://docs.scipy.org/doc/scipy/scipy-ref-1.2.1.pdf
from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import breadth_first_tree #arvore de busca em largura

X = csr_matrix([[0,8,0,3],[0,0,2,5],[0,0,0,6],[0,0,0,0]])

Tcsr=breadth_first_tree(X,0, directed=False)

Tcsr.toarray().astype(int)#matriz de adjascencia

print(Tcsr)#grafo dA Busca em largura

#%%

#https://www.tutorialspoint.com/scipy/
G_dense = np.array([ [0, 2, 1],
                     [2, 0, 0],
                     [1, 0, 0] ])
                     
G_masked = np.ma.masked_values(G_dense, 0)
from scipy.sparse import csr_matrix

G_sparse = csr_matrix(G_dense)
print(G_sparse.data)# G_sparse.data - pesos

'''Matplotlib

O matplotlib é uma biblioteca de plotagem de gráficos 2D . Ele pode ser usado como script com o ambiente interativo IPython,
Jupter Notebook, aplicações web services, entre outras. Os tipos de gráficos que podem ser plotados:
'''

#%%
# https://www.tutorialspoint.com/scipy/
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
x = np.linspace(0, 4, 12)
y = np.cos(x**2/3+4)

print(x,y)

plt.plot(x, y, 'o')
plt.show()



f1 = sp.interpolate.interp1d(x, y,kind = 'linear')

f2 = sp.interpolate.interp1d(x, y, kind = 'cubic')

print(f1)

xnew = np.linspace(0, 4,30)

plt.plot(x, y, 'o', xnew, f1(xnew), '-', xnew, f2(xnew), '--')

plt.legend(['data', 'linear', 'cubic','nearest'], loc = 'best')

plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.grid(True, linestyle='-.')

ax.tick_params(labelcolor='r', labelsize='medium', width=200)

#labelsize='medium', labelsize='large', labelsize=30

plt.show()




 # fazer superficies mpl_toolkits.mplot3d 

'''





import numpy as np
import matplotlib.pyplot as plt

def gaussiana(x):
    return np.arange(1000)*((1/np.sqrt(3.14))*(2.71828**(-(x-2)**2)))

v = gaussiana(0.6)
#plt.hist(v, bins=50, density=1) 
 
plt.show()
'''


'''

3 exemplo scipy



1 numpy
1matplotlib
1scipy

gráficos normais(x,y)
histogramas
grafico de barras
gráfico de erros
gráfico de dispersão



Numpy



Referências Bibliográficas

https://matplotlib.org/

https://www.scipy.org/about.html

https://www.numpy.org/








'''


#%%
