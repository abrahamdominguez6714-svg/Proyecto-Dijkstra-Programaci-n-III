"""
Nombre Completo: Abraham Dominguez Valdez 
Titulo: Proyecto deAlgoritmo de Dijkstra con Programación Orientada a Objetos
Fecha: 14/10/26
Materia: Programación III
Profesor: Diego Alberto Ayala Vazquez 

Descripción:
    Este programa implementa el algoritmo de Dijkstra para encontrar la ruta
    más corta entre dos nodos de un grafo ponderado y dirigido o no dirigido.
    El ejemplo representa una red de transporte entre estaciones, donde el
    peso de cada conexión son los minutos de viaje.

Clases:
    - Nodo:  representa un vértice del grafo (una estación).
    - Arista: representa una conexión con peso entre dos nodos.
    - Grafo: guarda los nodos y aristas, e implementa Dijkstra.

Idea general del algoritmo de Dijkstra:
    1. Asignar distancia 0 al nodo origen e infinito a todos los demás.
    2. Tomar siempre el nodo NO visitado con la menor distancia conocida.
    3. Para cada vecino de ese nodo, probar si pasar por él ofrece un camino
       más corto; si es así, actualizar la distancia y guardar de dónde
       venimos (nodo "anterior").
    4. Marcar el nodo como visitado y repetir hasta llegar al destino.
    5. Reconstruir la ruta caminando hacia atrás con los nodos "anteriores".

Restricción: Dijkstra solo funciona con pesos NO negativos.
"""

import heapq  # Módulo de Python que implementa una cola de prioridad (min-heap)


# ----------------------------------------------------------------------------
# Clase Nodo
# ----------------------------------------------------------------------------
class Nodo:
    """Representa un vértice (nodo) del grafo.

    Atributos:
        nombre (str): identificador único del nodo (ej. "A", "Centro").
    """

    def __init__(self, nombre):
        """Crea un nodo con el nombre indicado.

        Parámetros:
            nombre (str): nombre o etiqueta del nodo.
        """
        self.nombre = nombre

    def __str__(self):
        """Devuelve el nombre del nodo al imprimirlo con print()."""
        return self.nombre

    def __repr__(self):
        """Representación útil para depuración (se ve al imprimir listas)."""
        return f"Nodo({self.nombre!r})"


# ----------------------------------------------------------------------------
# Clase Arista
# ----------------------------------------------------------------------------
class Arista:
    """Representa una conexión con peso entre dos nodos.

    Atributos:
        origen (Nodo): nodo desde donde sale la conexión.
        destino (Nodo): nodo al que llega la conexión.
        peso (float): costo de recorrer la arista (distancia, tiempo, etc.).
    """

    def __init__(self, origen, destino, peso):
        """Crea una arista validando que el peso no sea negativo.

        Parámetros:
            origen (Nodo): nodo de salida.
            destino (Nodo): nodo de llegada.
            peso (float): costo de la conexión (debe ser >= 0).

        Lanza:
            ValueError: si el peso es negativo, porque Dijkstra no lo admite.
        """
        if peso < 0:
            raise ValueError("Dijkstra no admite pesos negativos.")
        self.origen = origen
        self.destino = destino
        self.peso = peso

    def __str__(self):
        """Devuelve la arista en formato legible: A --(4)--> B."""
        return f"{self.origen} --({self.peso})--> {self.destino}"


# ----------------------------------------------------------------------------
# Clase Grafo
# ----------------------------------------------------------------------------
class Grafo:
    """Grafo ponderado representado con una lista de adyacencia.

    La lista de adyacencia es un diccionario donde cada llave es el nombre de
    un nodo y su valor es la lista de aristas que salen de ese nodo.

    Ejemplo interno:
        {"A": [Arista(A,B,4), Arista(A,C,2)], "B": [...], ...}

    Atributos:
        nodos (dict): nombre -> objeto Nodo.
        adyacencia (dict): nombre -> lista de objetos Arista salientes.
    """

    def __init__(self):
        """Crea un grafo vacío (sin nodos ni aristas)."""
        self.nodos = {}
        self.adyacencia = {}

    # ------------------------- Construcción del grafo -----------------------
    def agregar_nodo(self, nombre):
        """Agrega un nodo al grafo si todavía no existe.

        Parámetros:
            nombre (str): nombre del nodo a agregar.

        Retorna:
            Nodo: el nodo creado (o el que ya existía).
        """
        if nombre not in self.nodos:
            self.nodos[nombre] = Nodo(nombre)
            self.adyacencia[nombre] = []
        return self.nodos[nombre]

    def agregar_arista(self, origen, destino, peso, dirigida=False):
        """Conecta dos nodos con una arista ponderada.

        Si alguno de los nodos no existe, se crea automáticamente.

        Parámetros:
            origen (str): nombre del nodo de salida.
            destino (str): nombre del nodo de llegada.
            peso (float): costo de la conexión.
            dirigida (bool): si es False (por defecto), la conexión funciona
                en ambos sentidos (se agrega también destino -> origen).
        """
        nodo_origen = self.agregar_nodo(origen)
        nodo_destino = self.agregar_nodo(destino)

        # Arista en el sentido normal: origen -> destino
        self.adyacencia[origen].append(Arista(nodo_origen, nodo_destino, peso))

        # Si el grafo NO es dirigido, también se puede ir de regreso
        if not dirigida:
            self.adyacencia[destino].append(
                Arista(nodo_destino, nodo_origen, peso)
            )

    # ------------------------------ Dijkstra --------------------------------
    def dijkstra(self, nombre_origen, nombre_destino):
        """Calcula la ruta más corta entre dos nodos con el algoritmo de Dijkstra.

        Parámetros:
            nombre_origen (str): nodo donde inicia el recorrido.
            nombre_destino (str): nodo donde termina el recorrido.

        Retorna:
            tuple: (distancia_total, ruta)
                - distancia_total (float): costo mínimo acumulado.
                - ruta (list[str]): nombres de los nodos en orden de recorrido.
                Si no existe camino, retorna (float("inf"), []).

        Lanza:
            ValueError: si el origen o el destino no existen en el grafo.
        """
        # --- Paso 0: validar que los nodos existan ---
        if nombre_origen not in self.nodos or nombre_destino not in self.nodos:
            raise ValueError("El nodo origen o destino no existe en el grafo.")

        # --- Paso 1: inicialización ---
        # Todas las distancias empiezan en infinito (aún no sabemos llegar)
        distancias = {nombre: float("inf") for nombre in self.nodos}
        # El origen está a distancia 0 de sí mismo
        distancias[nombre_origen] = 0

        # 'anterior' guarda desde qué nodo llegamos a cada nodo por el mejor
        # camino encontrado; sirve para reconstruir la ruta al final.
        anterior = {nombre: None for nombre in self.nodos}

        # Conjunto de nodos ya procesados (su distancia ya es definitiva)
        visitados = set()

        # Cola de prioridad: guarda tuplas (distancia, nombre_nodo).
        # heapq siempre entrega primero la tupla con la distancia más pequeña.
        cola = [(0, nombre_origen)]

        # --- Paso 2: ciclo principal ---
        while cola:
            # Sacamos el nodo con la menor distancia acumulada
            distancia_actual, nombre_actual = heapq.heappop(cola)

            # Si ya lo procesamos antes, esta entrada está desactualizada
            if nombre_actual in visitados:
                continue

            # Marcamos el nodo como visitado: su distancia ya es la mínima
            visitados.add(nombre_actual)

            # Si llegamos al destino, ya no hace falta seguir explorando
            if nombre_actual == nombre_destino:
                break

            # --- Paso 3: relajación de aristas ---
            # Revisamos cada vecino del nodo actual
            for arista in self.adyacencia[nombre_actual]:
                vecino = arista.destino.nombre

                # Ignoramos vecinos que ya fueron procesados
                if vecino in visitados:
                    continue

                # Distancia si llegamos al vecino pasando por el nodo actual
                nueva_distancia = distancia_actual + arista.peso

                # Si este camino es mejor que el que conocíamos, lo guardamos
                if nueva_distancia < distancias[vecino]:
                    distancias[vecino] = nueva_distancia
                    anterior[vecino] = nombre_actual
                    heapq.heappush(cola, (nueva_distancia, vecino))

        # --- Paso 4: verificar si el destino es alcanzable ---
        if distancias[nombre_destino] == float("inf"):
            return float("inf"), []

        # --- Paso 5: reconstruir la ruta de atrás hacia adelante ---
        ruta = []
        nodo = nombre_destino
        while nodo is not None:
            ruta.append(nodo)
            nodo = anterior[nodo]

        # La ruta quedó al revés (destino -> origen), así que la invertimos
        ruta.reverse()

        return distancias[nombre_destino], ruta

    # --------------------------- Utilidades de salida -----------------------
    def mostrar(self):
        """Imprime en pantalla todas las conexiones del grafo."""
        print("Conexiones del grafo:")
        for nombre, aristas in self.adyacencia.items():
            for arista in aristas:
                print(f"  {arista}")


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def construir_red_transporte():
    """Construye un grafo de ejemplo: una red de transporte entre estaciones.

    El peso de cada arista representa los minutos de viaje entre estaciones.

    Retorna:
        Grafo: el grafo ya cargado con nodos y conexiones.
    """
    grafo = Grafo()

    #                 origen      destino      minutos
    grafo.agregar_arista("Centro",   "Plaza",      4)
    grafo.agregar_arista("Centro",   "Mercado",    2)
    grafo.agregar_arista("Mercado",  "Plaza",      1)
    grafo.agregar_arista("Mercado",  "Terminal",   7)
    grafo.agregar_arista("Plaza",    "Terminal",   5)
    grafo.agregar_arista("Plaza",    "Hospital",   8)
    grafo.agregar_arista("Terminal", "Hospital",   3)
    grafo.agregar_arista("Terminal", "Aeropuerto", 6)
    grafo.agregar_arista("Hospital", "Aeropuerto", 2)

    return grafo


def main():
    """Función principal: arma el grafo, ejecuta Dijkstra y muestra resultados."""
    grafo = construir_red_transporte()
    grafo.mostrar()

    origen = "Centro"
    destino = "Aeropuerto"

    print(f"\nBuscando la ruta más corta de {origen} a {destino}...\n")

    distancia, ruta = grafo.dijkstra(origen, destino)

    if ruta:
        print("=== RESULTADO ===")
        print(f"Distancia/costo total: {distancia} minutos")
        print("Ruta:", " -> ".join(ruta))
    else:
        print(f"No existe un camino entre {origen} y {destino}.")


# Este bloque asegura que main() solo se ejecute cuando corremos este archivo
# directamente, y no cuando se importa como módulo desde otro programa.
if __name__ == "__main__":
    main()