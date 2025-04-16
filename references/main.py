# Importazione delle librerie necessarie
import gymnasium as gym         # Libreria per creare e gestire l'ambiente di reinforcement learning
import torch                    # Framework per la rete neurale e operazioni sui tensori
import torch.nn as nn           # Modulo di PyTorch per definire reti neurali
import torch.optim as optim     # Modulo di PyTorch per gli ottimizzatori
import numpy as np              # Libreria per la gestione di array e operazioni matematiche
import random                   # Modulo per generare numeri casuali
from collections import deque   # Struttura dati per implementare la replay memory
import matplotlib.pyplot as plt # Libreria per visualizzare i risultati

# Parametri globali per l'addestramento
EPISODES = 2000                 # Numero totale di episodi di addestramento
BATCH_SIZE = 64                 # Dimensione del batch per l'addestramento (aumentata per stabilità)
GAMMA = 0.99                    # Fattore di sconto per le ricompense future
EPSILON_START = 1.0             # Valore iniziale di epsilon per la politica epsilon-greedy
EPSILON_END = 0.02              # Valore finale di epsilon (leggermente più alto per esplorazione residua)
EPSILON_DECAY = 0.995           # Fattore di decadimento di epsilon per episodio
MEMORY_SIZE = 20000             # Dimensione della replay memory (aumentata per più esperienza)
LEARNING_RATE = 0.0005          # Tasso di apprendimento (ridotto per convergenza più stabile)
TARGET_UPDATE = 10              # Frequenza (in episodi) di aggiornamento della rete target
VISUALIZE_EVERY = 50            # Visualizza un episodio ogni 50 episodi

# Definizione della classe per la rete neurale DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()            # Inizializza la classe base nn.Module
        self.fc1 = nn.Linear(state_size, 256)  # Primo strato fully connected: input = stato, output = 256 neuroni
        self.fc2 = nn.Linear(256, 128)         # Secondo strato fully connected: input = 256, output = 128 neuroni
        self.fc3 = nn.Linear(128, action_size) # Strato di output: input = 128, output = numero di azioni

    def forward(self, x):           # Funzione che definisce il passaggio in avanti della rete
        x = torch.relu(self.fc1(x)) # Applica la funzione di attivazione ReLU al primo strato
        x = torch.relu(self.fc2(x)) # Applica ReLU al secondo strato
        x = self.fc3(x)             # Output lineare: restituisce i Q-values per ogni azione
        return x                    # Ritorna i Q-values

# Classe per la replay memory
class ReplayMemory:
    def __init__(self, capacity):   # Inizializzatore della memoria
        self.memory = deque(maxlen=capacity)  # Crea una deque con capacità massima specificata

    def push(self, transition):     # Metodo per aggiungere una transizione alla memoria
        self.memory.append(transition)  # Aggiunge la tupla (state, action, reward, next_state, done)

    def sample(self, batch_size):   # Metodo per estrarre un batch casuale dalla memoria
        return random.sample(self.memory, batch_size)  # Restituisce una lista di batch_size transizioni casuali

    def __len__(self):              # Metodo per ottenere la lunghezza attuale della memoria
        return len(self.memory)     # Ritorna il numero di transizioni memorizzate

# Funzione per selezionare un'azione con la politica epsilon-greedy
def select_action(state, epsilon, model, action_size):
    if random.random() > epsilon:   # Se un numero casuale è maggiore di epsilon, sfrutta (exploit)
        with torch.no_grad():       # Disabilita il calcolo dei gradienti per risparmiare memoria
            q_values = model(state) # Calcola i Q-values per lo stato corrente
            return q_values.argmax().item()  # Restituisce l'indice dell'azione con Q-value massimo
    else:                           # Altrimenti, esplora (explore)
        return random.randrange(action_size)  # Restituisce un'azione casuale tra 0 e action_size-1

# Funzione per ottimizzare il modello
def optimize_model(model, target_model, memory, optimizer):
    if len(memory) < BATCH_SIZE:    # Se la memoria non ha abbastanza transizioni, non fare nulla
        return
    transitions = memory.sample(BATCH_SIZE)  # Estrai un batch casuale di transizioni

    # Estrai componenti dalle transizioni
    batch_state = torch.cat([t[0] for t in transitions])  # [batch_size, state_size]
    batch_action = torch.LongTensor([t[1] for t in transitions]).view(-1, 1)  # [batch_size, 1]
    batch_reward = torch.FloatTensor([t[2] for t in transitions])  # [batch_size]
    batch_next_state = torch.cat([t[3] for t in transitions])  # [batch_size, state_size]
    batch_done = torch.FloatTensor([t[4] for t in transitions])  # [batch_size]

    # Calcola i Q-values attuali per le azioni eseguite
    current_q_values = model(batch_state).gather(1, batch_action).squeeze(1)  # [batch_size]

    # Calcola i Q-values target usando la rete target
    with torch.no_grad():           # Disabilita il calcolo dei gradienti per i target
        next_q_values = target_model(batch_next_state).max(1)[0]  # Q-values massimi degli stati successivi
        target_q_values = batch_reward + (GAMMA * next_q_values * (1 - batch_done))  # Formula di Bellman

    # Calcola la loss (errore quadratico medio)
    loss = nn.MSELoss()(current_q_values, target_q_values)

    # Esegui l'ottimizzazione
    optimizer.zero_grad()           # Resetta i gradienti dell'ottimizzatore
    loss.backward()                 # Calcola i gradienti della loss rispetto ai parametri del modello
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Limita la norma dei gradienti per stabilità
    optimizer.step()                # Aggiorna i pesi del modello con l'ottimizzatore

# Funzione per visualizzare un episodio
def visualize_episode(model, episode):
    env = gym.make("CartPole-v1", render_mode="human")  # Crea un ambiente separato per la visualizzazione
    print(f"Visualizzazione episodio {episode+1}")  # Stampa il numero dell'episodio
    state, _ = env.reset()  # Resetta l'ambiente
    state = torch.FloatTensor(state).unsqueeze(0)  # Converte lo stato in tensore [1, state_size]
    done = False  # Flag di terminazione
    total_reward = 0  # Ricompensa totale per l'episodio visualizzato
    while not done:
        with torch.no_grad():  # Disabilita i gradienti per la visualizzazione
            q_values = model(state)  # Calcola i Q-values
            action = q_values.argmax().item()  # Scegli l'azione con Q-value massimo
        next_state, reward, done, _, _ = env.step(action)  # Esegui l'azione
        state = torch.FloatTensor(next_state).unsqueeze(0)  # Aggiorna lo stato
        total_reward += reward  # Accumula la ricompensa
    print(f"Ricompensa episodio visualizzato: {total_reward}")  # Stampa la ricompensa
    env.close()

# Inizializzazione dell'ambiente e delle reti
env = gym.make("CartPole-v1")  # Crea l'ambiente con rendering abilitato
state_size = env.observation_space.shape[0]  # Dimensione dello spazio di stato (4 per CartPole)
action_size = env.action_space.n    # Numero di azioni possibili (2 per CartPole)

model = DQN(state_size, action_size)       # Inizializza la rete principale (policy network)
target_model = DQN(state_size, action_size)  # Inizializza la rete target
target_model.load_state_dict(model.state_dict())  # Copia i pesi iniziali dalla rete principale
target_model.eval()                 # Imposta la rete target in modalità valutazione (no training)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Ottimizzatore Adam per la rete principale
memory = ReplayMemory(MEMORY_SIZE)  # Inizializza la replay memory

# Variabili per il tracciamento
epsilon = EPSILON_START             # Inizializza epsilon al valore iniziale
episode_rewards = []                # Lista per salvare le ricompense di ogni episodio

# Ciclo principale di addestramento
for episode in range(EPISODES):
    state, _ = env.reset()          # Resetta l'ambiente e ottiene lo stato iniziale
    state = torch.FloatTensor(state).unsqueeze(0)  # Converte lo stato in tensore e aggiunge una dimensione batch
    total_reward = 0                # Inizializza la ricompensa totale per l'episodio
    done = False                    # Flag per indicare se l'episodio è terminato
    
    while not done:                 # Finché l'episodio non è terminato
        action = select_action(state, epsilon, model, action_size)  # Seleziona un'azione
        next_state, reward, done, _, _ = env.step(action)  # Esegui l'azione nell'ambiente
        next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Converte il prossimo stato in tensore
        total_reward += reward      # Aggiorna la ricompensa totale

        # Crea una transizione e aggiungila alla memoria
        transition = (state, action, reward, next_state, done)
        memory.push(transition)     # Memorizza la transizione
        state = next_state          # Aggiorna lo stato corrente al prossimo stato
        optimize_model(model, target_model, memory, optimizer)  # Ottimizza il modello

    # Aggiorna la rete target ogni TARGET_UPDATE episodi
    if episode % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())  # Copia i pesi dalla rete principale alla rete target

    # Decadimento di epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)  # Riduci epsilon, ma non sotto EPSILON_END

    # Salva e stampa i risultati dell'episodio
    episode_rewards.append(total_reward)
    print(f"Episodio {episode+1}/{EPISODES} | Ricompensa: {total_reward} | Epsilon: {epsilon:.3f}")

    # Visualizza l'episodio ogni VISUALIZE_EVERY episodi
    if (episode + 1) % VISUALIZE_EVERY == 0:
        visualize_episode(model, episode)

# Chiudi l'ambiente
env.close()

# Visualizza i risultati
plt.plot(episode_rewards)           # Plotta le ricompense per episodio
plt.title("Ricompense per Episodio")  # Titolo del grafico
plt.xlabel("Episodio")              # Etichetta dell'asse X
plt.ylabel("Ricompensa Totale")     # Etichetta dell'asse Y
plt.grid(True)                      # Aggiunge una griglia per leggibilità
plt.show()                          # Mostra il grafico