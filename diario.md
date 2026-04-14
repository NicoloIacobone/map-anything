13/04/2026
Oggi ho ripreso a lavorare al progetto, ricontrollando un pò il codice e lanciando alcune run di test per vedere se è tutto ok.
In particolare sono stati testati i dataloader e il processo di save/resume checkpoint.
Al momento ho lanciato una distillazione da 1000 epoche senza usare la consistency loss per verificare se e quanto le features dello student sono coerenti tra loro.
Ho anche provato a reimplementare tutto su VGGT con l'idea di usare SAM3, ma non ne vale la pena.

Modifiche:
- Ridotta la risoluzione di training da 512 a 224 per velocizzare i test. Aspect ratio 1:1.

Cose imparate:
- Per quanto riguarda la risoluzione, SAM teacher fa sempre Resize((resolution, resolution)) e viene istanziato con resolution=1024, quindi bisogna usare sempre risoluzione quadrata del dataset, altrimenti bisogna capire come gestire aspect ratios diversi.
- Negli yaml dei dataset, la dimensione del dataset viene definita come n @ {...} dove n è il numero di volte in cui viene campionata una scena, e {...} è la configurazione della scena. Per esempio 50 @ {...} se è stato impostato n_views=4, significa che vengono campionate 50 scene, ognuna da 4 views, circa 200 immagini totali.

TODO LIST:
- Controllare coerenza senza consistency loss
- Rieseguire lo stesso test con consistency loss e controllare coerenza
- Implementare grouping con HDBSCAN
- Implementare validazione per ottenere metriche quantitative di coerenza e segmentazione
- Implementare decoder D4RT in versione semantic segmentation

14/04/2026
Ho analizzato i risultati della distillazione con e senza consistency loss, e mi sembrano uguali.
Il problema è che ho fatto un resume ma lasciando la configurazione della loss con entrambe le loss, quindi il risultato finale è lo stesso.
Faccio ripartire solo la distillazione con distillation loss.
Per evitare di confondermi ho splittato in 3 le configurazioni della loss, in modo da essere più chiaro su cosa viene usato in ogni run.

Cose imparate:
- Il cluster in single GPU con stessa configurazione del locale è circa 7 volte più veloce (3H vs 21H), quindi per test non va mai usato il locale.