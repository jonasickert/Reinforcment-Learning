## 28.06

    - Lokale Implementierung der Wrapper Methoden um sich mit Jonas austauschen zu koennen 
    - Implementierung hat es nicht ins Git geschafft da Ansatz von Jonas besser

## 01.07

    meeting mit Jonas über Aufgabe 1 (2h)
    - verschiedene Lösungsansätze 
    - wie FrameStacking umsetzen
    - Oberservation aus meiner Implementation uebernommen
    - step und reset durch observation getauscht, preprocessing obsolete

## 04.07

    - Implementierung der Netzwerk Struktur fuer Features & Pixels:
        Freatures:
            - Input Layer
            - 2 Hidden Layer mit jeweils 128 neuronen
            - Output layer abhaenging von den moeglichen Aktionen
            - ReLu als Activation Function benutzt
        
        Pixels:
            - 3 CNN Layer
            - Kernel Size  8x8, 4x4, 3x3
            - Stride 4, 2, 1
            - Fully connected layer mit 512px fuer 84x84 image
            - ReLu als Activation Function Benutzt
    
    - Documentation Hinzugefuegt

## 09.07

    - Wandb Account erstellt
    - wandb schnittstelle implementiert
    - Evaluate um args erweitert um logs optional auf wandb anzeigen zu lassen
    - Video um args erweitert um logs optional auf wandb anzeigen zu lassen


