# dueling-architectures

## Literatura

Originalan rad: [Dueling architectures](https://arxiv.org/pdf/1511.06581.pdf) 

Napomena: Moj model ne koristi prioritizovanu replay memoriju

## Rezultati

### Prvi model

U direktorijumu models1 nalaze se parametri neuronske mreže nakon svakih 100 odigranih igara u fazi treniranja. U direktorijumu scores nalaze se srednje vrednosti nagrada ostvarenih kroz svakih 100 odigranih igara u fazi treniranja, kao i informacija o najboljim ostvarenim rezultatima.

Ono što je karakteristično za prvi model je:
- Maksimalna veličina replay memorije: 700000 frejmova
- Learning rate Adam optimajzera: 0.00001
- Loss funkcija: mean squared error
- Ukupan broj frejmova (dužina treniranja): 2.5 miliona frejmova, što je oko 40 sati treniranja na mojoj mašini
