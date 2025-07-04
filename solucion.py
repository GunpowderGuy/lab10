"""
Solución para Lab 10: Reinforcement Learning
MDP del robot mensajero en un pasillo de 4 casillas (0–3),
implementación de Value Iteration y derivación de la política óptima.
Basado en la práctica descrita en el notebook Lab_10_Reinforcement_Learning.ipynb :contentReference[oaicite:0]{index=0}.
"""

import numpy as np
from tabulate import tabulate
from pprint import pprint

# 1️⃣ Definición del MDP

states = [0, 1, 2, 3]
terminal_states = [3]
actions = ['avanza', 'retrocede', 'espera']

def build_P():
    """
    Construye el diccionario P[s][a] = [(p, s', r), ...]
    con las transiciones estocásticas descritas en la práctica.
    """
    P = {s: {a: [] for a in actions} for s in states}
    for s in states:
        for a in actions:
            # Estado terminal: permanece con recompensa 0
            if s in terminal_states:
                P[s][a] = [(1.0, s, 0.0)]
                continue

            # Acción con éxito (prob 0.9)
            if a == 'avanza':
                s_next = min(s + 1, 3)
            elif a == 'retrocede':
                s_next = max(s - 1, 0)
            else:  # espera
                s_next = s

            # Recompensa: +1 si llega a 3, -0.1 en otro caso
            r = 1.0 if s_next == 3 else -0.1
            P[s][a].append((0.9, s_next, r))

            # Fallo de motor (prob 0.1): se queda en s sin coste extra
            P[s][a].append((0.1, s, 0.0))

    return P

P = build_P()

if __name__ == '__main__':
    # Sanity-check: muestra las transiciones para (s=0, a="avanza")
    print("Ejemplo de P[0]['avanza']:")
    pprint(P[0]['avanza'])

    # 2️⃣ Value Iteration
    gamma = 0.9     # factor de descuento
    theta = 1e-6    # umbral de convergencia
    V = {s: 0.0 for s in states}

    iteration = 0
    while True:
        delta = 0.0
        for s in states:
            if s in terminal_states:
                continue
            v_old = V[s]
            # Calcula Q(s,a) para cada acción
            q_values = [
                sum(p * (r + gamma * V[s2]) for p, s2, r in P[s][a])
                for a in actions
            ]
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        iteration += 1
        if delta < theta:
            break

    print(f"\nConverged in {iteration} iterations\n")
    print(tabulate([[s, round(V[s], 4)] for s in states],
                   headers=['Estado', 'V*(s)']))

    # 3️⃣ Derivación de la política óptima
    policy = {}
    for s in states:
        if s in terminal_states:
            policy[s] = None
        else:
            # Escoge la acción con mayor Q(s,a)
            best_action = max(
                actions,
                key=lambda a: sum(p * (r + gamma * V[s2]) for p, s2, r in P[s][a])
            )
            policy[s] = best_action

    print("\nPolítica óptima π*:")
    print(tabulate([[s, policy[s]] for s in states],
                   headers=['Estado', 'π*(s)']))
