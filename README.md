# CartPole Deep Reinforcement Learning

Questo progetto implementa un agente DQN per risolvere il gioco CartPole utilizzando Python, PyTorch e Gymnasium.


## Installazione
1. Clona il repository:
   git clone https://github.com/tuo-username/cartpole-drl-project.git
   cd cartpole-drl-project

2. Installa le dipendenze:
   pip install -r requirements.txt


## Esecuzione
Per addestrare l'agente:
    python run.py --train

Per riprendere l'addestramento da un checkpoint:
    python run.py --train --resume

Per visualizzare un episodio:
    python run.py --visualize


## Struttura del progetto
src/: Codice sorgente.

config/: File di configurazione.

data/: Risultati generati.

tests/: Test unitari.


## Licenza
MIT License


![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

