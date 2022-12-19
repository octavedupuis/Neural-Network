import numpy as np
import matplotlib.pyplot as plt

Target = np.array([ [1],   [0],   [1],   [0], [0]])
def sigmoid (x): return 1/(1 + np.exp(-x))              # fonction d'activation
def sigmoid_(x): return np.exp(-x)*((sigmoid(x))**2)    # dérivée de la fonction d'activation

def prod_tat(L1,L2):
    L = np.zeros(np.shape(L1))
    for k in range (len(L1)):
        L[k,0] = L1[k,0]*L2[k,0]
    return L

Xtotal = np.array([[1,1,1,0], [0,1,0,0], [1,1,1,1], [0,0,0,1], [0,0,0,0]])
Target = np.array([ [1],   [0],   [1],   [0], [0]])

def NN(epochs, alpha=.1, fun_act=sigmoid, fun_act_=sigmoid_, Xtotal = Xtotal, target = Target):# epochs correspond au nombre d'itérations, alpha au learning rate, fun_act à la fonction d'activation et fun_act_ à sa dérivée
    
    inputLayerSize, hiddenLayerSize, outputLayerSize = np.shape(Xtotal)[1], 3, 1
    np.random.seed(1234) 
    
    n = np.shape(Xtotal)[0] # Nombre de dimension          
                       
    V = np.random.uniform(size=(inputLayerSize,hiddenLayerSize))    # poids de l'entrée
    W = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))   # poids de la couche cachée
          
    Vbiais = np.random.uniform(size=(1,hiddenLayerSize)).T
    Wbiais = np.random.uniform(size=(1,outputLayerSize)).T
          
    Yout = np.zeros((np.shape(Target)))
          
    for e in range(epochs):
        for ipt in range(n): # On parcours chacunes des données disponibles
        
            X = np.array([Xtotal[ipt,:]]).T
        
            #-------------------- FeedForward -------------------- 
            Zin = np.dot(X.T,V).T + Vbiais      # On calcul la somme pondérée des entrée vers la couche cachée en ajoutant le biais
            Z = fun_act(Zin)                    # On applique à ces sommes pondérées la fonction d'activation
            Yin = np.dot(Z.T, W).T + Wbiais     # On calcul pour la sortie la somme pondérée des valeurs issus de la couche cachée, en ajoutant le biais
            
            #-------------------- Prédictions -------------------- 
            Y = int(2*fun_act(Yin))             # On calcul les prédictions, en appliquant d'abord la fonction d'activation puis en associant 0 ou 1 en fonction de si la valeur est au dessus ou en dessous de 0.5
              
            #-------------------- BackPropagation -------------------- 
            # En partant de la sortie, on compare les préditions avec les targets, et on calcul l'erreur associée
            deltak = (Target[ipt,0] - Y)*fun_act_(Yin)
            # On peut ainsi calculer le terme de correction des poids et du biais de la couche cachée 
            deltaW = alpha*deltak*Z # Correction des poids
            deltaWB = alpha*deltak  # Correction du biais
              
            # On se place au niveau de la couche cachée, et on somme les erreurs de la couche du dessus en les pondérant par les poids
            delta_in_j = deltak*W
            # On calcul le terme d'erreur
            deltaj = prod_tat(delta_in_j, Zin)
            # On peut ainsi calculer le terme de correction des poids et du biais de l'entrée
            deltaV = np.dot(X,deltaj.T)     # Correction des poids
            deltaVB = alpha*deltaj          # Correction du biais
              
            #-------------------- Mise à jour -------------------- 
            W +=  deltaW
            Wbiais += deltaWB
              
            V += deltaV 
            Vbiais += deltaVB
              
            Yout[ipt,0] = Y # On stocke les prédictions faites
    return(Yout)

def acc(Yout):
    c = 0
    for i in range(len(Yout)):
        if Yout[i,0]==Target[i,0]:
            c+=1
    return c/len(Yout)
    
epochs = [k for k in range (1,20)]
learning_rate = np.linspace(0.05,5,7)
for alpha in learning_rate :
    stat = []
    for k in epochs:
        cv = []
        for i in range (2):
            cv.append(acc(NN(k,alpha)))
        stat.append(np.mean(cv))
    plt.plot(epochs, stat, label = 'Learning rate = %g'%alpha)

plt.legend()
plt.ylabel('Pourcentage de justesse')
plt.xlabel('Epochs')
plt.title('Réseau de Neurones "from Scratch"')



