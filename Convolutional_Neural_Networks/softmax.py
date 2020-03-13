"""
numpy: maak gebruik van grondgetal 'e'
matplotlib: grafiek plotten van de softmax curve
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Om uiteindelijk een totaal plaatje te krijgen van een CNN 
moet het programma ook voorspellingen kunnen maken aan de 
hand van kansen. Dit wordt gedaan door de laatste layer genaamd 
de soft max. Dit is een dense layer (volledig verbonden) wat gebruik 
maakt van de Softmax functie, ook wel activatie functie genoemd zoals 
je die misschien wel kent als Sigmoid, Tanh of ReLu.
"""

def softmax_functie(logits):
    """
    De volgende stappen worden ondernomen:
    	1) e tot de macht n, waarin n elk gegeven output node is.
		2) tel al deze uitkomsten bij elkaar op: denominator
		3) elk uitkomst bij stap 1 wordt gebruikt als numerator
		4) kans: numerator / denominator 

        Voorbeeld: 
        1)  gegeven outputs: [0.5, 1, 3, 4]
            e^0.5 + e^1 + .. e^n = 79.0506 = denominator
            kans per output = e^0.5 / denominator = 0.02085651 (2%)
    """
    # return np.exp(logits) / sum(np.exp(logits)) dit kan in 1 line maar om het even duidelijker te laten zien:
    numerator = np.exp(logits)
    denominator = sum(numerator)

    return numerator / denominator

"""Geef hier de lijst op met de outputs van de nodes"""
logits = np.array([0.5,1,3,4])
# print(sum(np.exp(logits)))

"""Check hier of het totaal van de uitkomsten 1 is"""
probabilities = sum(softmax_functie(logits))

print(softmax_functie(logits)) #[0.00928666 0.04161993 0.11313471 0.8359587]
print(probabilities)

"""Plot de Softmax curves"""
def plot_me(logits):
    scores = np.vstack([logits, np.ones_like(logits), 0.2 * np.ones_like(logits)])
    plt.plot(logits,softmax_functie(logits).T, linewidth=2)
    plt.show()

# plot_me(logits)