## 28/04/2026
Testato il dataset completo su workspace.
Zippato e copiato su /work come backup e copiato su /scratch per essere utilizzato sul nodo.
Creato e testato script per copiare da /scratch alla cartella temporanea $TMPDIR sul nodo che non ha limiti di spazio o numero di file (gestito da parametro --tmp).

Individuato un mismatch logico tra train e inference: in fase di train faccio interpolazione per ridurre la shape delle informazioni geometriche a quelle semantiche es 224x224 -> 64x64, mentre in fase di inference interpolo le features semantiche per matchare shape immagini 64x64 -> 224x224.

Cose imparate:
- Quando avvio un job slurm viene creata una cartella temporanea sotto /scratch/tmp.[numero_job].niacobone che è diversa da /cluster/scratch/niacobone/tmp.

Idea: usare una loss leggermente diversa, più simile a una contrastive (InfoNCE) ma con le seguenti caratteristiche (check):
- Il check della semantica avviene solo nei punti coperti da una maschera valida prodotta da SAM2
- Aumentare il peso del negativo in base alla distanza tra i due punti presi (es. lungo il raggio di una circonferenza) in quanto teoricamente due punti distanti appartengono a due oggetti separati.
- Aumentare il peso del negativo in base al valore delle features es. delta(feat_target_1, feat_target_2) in quanto presuppongo che SAM2 produca features simili per oggetti simili quindi se le features sono diverse può essere che gli oggetti siano diversi.
- Definizione standard di InfoNCE per separare le features di punti diversi (sembra controintuitivo perché la loss è minimizzata se due punti dello stesso oggetto hanno features contrastanti, ma comunque va considerato che c'è anche la feature loss con un peso maggiore).


TODO aggiornata:
- [x] Dataset funzionante
    - [x] Zippare dataset completo su pf-pc20
    - [x] Copiarlo su cluster sotto /work/igp_psr/niacobone/distillation/dataset
    - [x] Copiarlo su cluster sotto /scratch/niacobone/distillation/dataset
    - [x] Creare script copia dataset in $TMP
- [ ] Script SAM2 mask multi-view
    - [ ] Usare la video mode di SAM2 per produrre maschere di segmentazione coerenti per ogni frame
    - [ ] Fare reprojection delle maschere sulla pointcloud generata da MapAnything
- [ ] Lettura papers
    - [ ] D4RT
    - [x] DETR
    - [ ] AnyRecon
    - [ ] GenReg
    - [ ] SimCLR
- [ ] Creare un quantitative test coerente
- [ ] Sistemare HDBSCAN sulla versione di test di MapAnything
    - Probabilmente non va sistemato, ma fallisce in quanto la nube di punti della feature pointcloud è troppo poco densa per poter essere facilmente clusterizzata.
- [ ] Capire il ruolo della confidence e come viene addestrata

## 27/04/2026
TODO:
- [ ] Dataset funzionante
    - [ ] Zippare dataset completo su pf-pc20
    - [ ] Copiarlo su cluster sotto /work/igp_psr/niacobone/distillation/dataset
    - [ ] Creare script copia dataset in $TMP
- [ ] Script SAM2 mask multi-view
    - [ ] Usare la video mode di SAM2 per produrre maschere di segmentazione coerenti per ogni frame
    - [ ] Fare reprojection delle maschere sulla pointcloud generata da MapAnything
- [ ] Lettura papers
    - [ ] D4RT
    - [x] DETR
    - [ ] AnyRecon
    - [ ] GenReg
    - [ ] SimCLR
- [ ] Creare un quantitative test coerente
- [ ] Sistemare HDBSCAN sulla versione di test di MapAnything


## 16/04/2026
Oggi preparo le visualizzazioni per il meeting (di domani).
Ho raggruppato alcune scene che voglio utilizzare per la segmentation e che sono parte del dataset su cui ho fatto l'overfit.
Oggi cerco anche di settare al meglio i threshold di confidence di produzione della pointcloud e i parametri di HDBSCAN per ottenere una segmentazione più pulita possibile.
Bisogna modificare il funzionamento di HDBSCAN in quanto al momento clusterizza in single-view e non in multi-view.

Ci sono due varianti pratiche:
- Variante minima: concateni tutte le feature di tutte le view in un unico array e fai HDBSCAN una sola volta. Questo è semplice, ma non sfrutta davvero la geometria multi-view.
- Variante completa: fuse per corrispondenza 3D, poi cluster globale. Questa è quella coerente con il training che hai descritto, perché la consistency loss ha già insegnato al modello a rendere simili le feature dello stesso punto 3D attraverso viste diverse.

La consistency loss consente di avere coerenza per la semantica dello stesso punto visto da diverse viste, ma non sa quando due punti appartengono allo stesso oggetto. Me ne rendo conto vedendo come il fronte e il retro della rappresentazione della statuetta hanno colori completamente diversi, nonostante siano vicini nello spazio 3D e facciano parte dello stesso oggetto.

    ```bash
    rsync -a --info=progress2 --partial -v blendedmvs converted/mapanything_dataset_metadata converted/wai_data/blendedmvs niacobone@euler.ethz.ch:/cluster/scratch/niacobone/distillation/dataset
    ```


## 15/04/2026
Ho analizzato i risultati della distillazione con solo distillation loss, il problema persiste, ma credo sia dovuto ad un mio errore: ho trainato solo su 50 immagini e testato su 2, quindi è più un overfitting, e avrei dovuto visualizzare le features del teacher e dello student su quelle stesse immagini.
Ho anche iniziato la copia del dataset su cluster, per il quale poi eseguirò il codice per sistemare i symlinks e poi lo passerò sulla partizione /work/igp_psr/niacobone/distillation/dataset per tenerlo backuppato.

Test effettuati:
- Resume della distillazione precedente ma con overfit = true per visualizzare le features sulle immagini su cui è stato fatto l'overfit.

Cose imparate:
- Devo stare attento a non usare il dataset di test quando quello di train è molto piccolo, altrimenti è facile utilizzare dati non "corretti" per trarre conclusioni.
- Il backward avviene dopo l'analisi di ogni batch, quindi più volte per epoca. In particolare avviene ogni max_imgs_per_gpu / num_views scene, quindi se max_imgs_per_gpu = 8 e num_views = 4, avviene ogni 2 scene.

TODO LIST:
- [x] Lanciare una run multi-view solo con distillation loss e overfit = true, per capire quanto sono coerenti le features della stessa scena.
    - [x] distillation_loss_full_dataset: run inizializzata da 0 con 4 views per scena --> le features sembrano abbastanza coerenti tra loro.
    - [x] test_distillation_loss_new_resume_multiview: run resumed da train con 1 view per scena --> le features sembrano abbastanza coerenti tra loro. Nessuna differenza visibile tra le due run.
    - [ ] Eseguire lo script di visualizzazione + HDBSCAN per vedere se la segmentazione è abbastanza buona.
    - [x] resume con consistency loss a 0.1 - resume_1_consistency_01 - ad occhio le features sembrano coerenti
    - [x] resume con consistency loss a 0.5 - resume_2_consistency_05 - ad occhio le features sembrano coerenti
    - [x] resume con consistency loss a 1.0 - resume_3_consistency_1 - le features non vanno per nulla bene in quanto non c'è più supervisione del teacher e le student features diventano uniformi
- [ ] Lanciare una run multi-view con distillation loss + consistency loss (0.1 peso) per capire come influisce.
- [ ] Da dentro /scratch2/nico/distillation/dataset eseguire il comando per copiare il dataset su cluster:
    ```bash
    rsync -a --info=progress2 --partial blendedmvs converted/mapanything_dataset_metadata converted/wai_data/blendedmvs niacobone@euler.ethz.ch:/cluster/scratch/niacobone/distillation/dataset
    ``` 
    e poi runnare lo script per sistemare i symlinks (vedi chat "correzione dei collegamenti delle immagini nel preprocessing").
    - [ ] Step 1: Test in dry-run (sicuro, non modifica nulla)
    ```bash
    rsync -a --info=progress2 --partial --dry-run blendedmvs converted/mapanything_dataset_metadata converted/wai_data/blendedmvs niacobone@euler.ethz.ch:/cluster/scratch/niacobone/distillation/dataset
    ```
    - [ ] Step 2: Se il dry-run è ok, eseguire il comando vero (con verbose per vedere ogni operazione)


## 14/04/2026
Ho analizzato i risultati della distillazione con e senza consistency loss, e mi sembrano uguali.
Il problema è che ho fatto un resume ma lasciando la configurazione della loss con entrambe le loss, quindi il risultato finale è lo stesso.
Faccio ripartire solo la distillazione con distillation loss.
Per evitare di confondermi ho splittato in 3 le configurazioni della loss, in modo da essere più chiaro su cosa viene usato in ogni run.

Cose imparate:
- Il cluster in single GPU con stessa configurazione del locale è circa 7 volte più veloce (3H vs 21H), quindi per test non va mai usato il locale.
- Quando vengono modificati i parametri del dataset (max_num_of_imgs_per_gpu, num_views, ecc) è necessario controllare che il numero di immagini totali per scena sia sufficiente per il batch size e il numero di epoche, altrimenti va in errore.
- Il probema delle features strane non deriva dalla risoluzione

Cose da risolvere:
- Capire perché la distillazione solo con distillation loss fa generare una strana noise visualizzabile con PCA dello student, nonostante la loss sia bassa. Ipotizzo centri qualcosa con la fusione delle features per scena.

Modifiche:
- Rimosso lo scaling della distillation loss in quanto deve essere 1:1 tra teacher e student, Lasciato invariato lo scaling per la consistency loss.

Test in corso:
- Distillazione solo con distillation loss e num_view = 1, per capire se il problema del noise è legato ad un numero di views maggiore di 1.


## 13/04/2026
Oggi ho ripreso a lavorare al progetto, ricontrollando un pò il codice e lanciando alcune run di test per vedere se è tutto ok.
In particolare sono stati testati i dataloader e il processo di save/resume checkpoint.
Al momento ho lanciato una distillazione da 1000 epoche senza usare la consistency loss per verificare se e quanto le features dello student sono coerenti tra loro.
Ho anche provato a reimplementare tutto su VGGT con l'idea di usare SAM3, ma non ne vale la pena.

Modifiche:
- Ridotta la risoluzione di training da 512 a 224 per velocizzare i test. Aspect ratio 1:1.

Cose imparate:
- Per quanto riguarda la risoluzione, SAM teacher fa sempre Resize((resolution, resolution)) e viene istanziato con resolution=1024, quindi bisogna usare sempre risoluzione quadrata del dataset, altrimenti bisogna capire come gestire aspect ratios diversi.
- Negli yaml dei dataset, la dimensione è definita come n @ {...}, dove n è il numero di sample-scena per epoca (non il numero di immagini), e {...} è la configurazione del dataset: per esempio, con n_views=4, 50 @ {...} significa 50 scene campionate per epoca (con possibili ripetizioni se le scene reali sono meno di 50), cioè circa 200 immagini totali processate nell’epoca.

TODO LIST:
- [x] Controllare coerenza senza consistency loss
- [x] Rieseguire lo stesso test con consistency loss e controllare coerenza
- [ ] Implementare grouping con HDBSCAN
- [ ] Implementare validazione per ottenere metriche quantitative di coerenza e segmentazione
- [ ] Implementare decoder D4RT in versione semantic segmentation