4.1 Datasets: Quali dataset hai usato (es. ScanNet, Replica, o dataset custom multi-view).
4.2 Implementation Details: Hardware, framework (PyTorch), iperparametri, strategie di training, gestione della memoria.
4.3 Evaluation Metrics: Come misuri il successo della distillazione (es. Feature distance, coerenza multi-vista, metriche proxy di segmentazione se applicabili).

We first validated the pipeline via overfitting. Here we visualize the learned features using PCA. Starting from random noise, you can see how the DPT head progressively learns to segment instances, eventually producing features that are visually very similar to the SAM teacher, both on COCO and BlendedMVS scenes.
To verify the quality of the learned latent space, we performed a 'Consistency Test'. We fed our distilled student features into the frozen SAM Decoder. As you can see, the resulting mask on the left is nearly identical to the ground truth on the right. This proves that the Student has successfully learned a valid representation of the Teacher's latent space.
Moving to full dataset training, we observe that the network generalizes well on unseen images and scenes. However, we do notice a slight degradation in boundary sharpness compared to the overfitting scenario.

Furthermore, we can observe distinct architectural artifacts in the Teacher's features. Interestingly, the Student fails to replicate these artifacts. This suggests that these patterns are structural byproducts of the Teacher's specific architecture which the Student cannot simply reproduce through imitation, highlighting a fundamental difference in how the two models encode spatial information.
Analyzing the loss curves, we encounter a plateau. This indicates a capacity bottleneck. We are attempting to compress the knowledge of the SAM 2 Image Encoder, a massive Hierarchical Transformer, into a lightweight DPT head. From an Information Bottleneck perspective, the student head simply lacks the capacity to fully represent the teacher's complex manifold without adapting the backbone.

In this first example, we compare two runs with identical settings, differing only in the weighting of the loss components.

In this second example we compare again two identical runs, but where the dotted line appears, the LR drops by a factor of 10.

This demonstrates that once the DPT head reaches its saturation point, dropping the learning rate yields no improvement. The bottleneck is structural, not optimization-related.

This led to our key insight regarding the 'Unfreeze Strategy'. This proved to be the only strategy capable of breaking through the loss plateau.
In purple, unfreezing only the 3D Transformer yields minimal gains. However, in orange, unfreezing the DINOv2 Encoder results in a drastic drop in loss. This demonstrates that for fine-grained semantic tasks, the geometric backbone itself must be fine-tuned to support the high-frequency details required by SAM.

4.1 Datasets:
- descrizione dataset coco (SV)
- descrizione dataset blendedmvs (MV)
- pre-processing blendedmvs con wai-processing di MapAnything (conversion, covisibility, esecuzione modello MOGE per depth, mask, normals)
- motivazione scelta dataset (coco per validazione pipeline, blendedmvs per test distillazione multi-view e consistency loss)
- motivazione produzione depth, mask, normals anche se non necessari per obiettivo tesi (dataset più completo e versatile per futuri esperimenti con finetuning backbone)

Sono stati utilizzati due dataset: COCO2017 e BlendedMVS.
[Descrizione di entrambi i dataset, numero di immagini, tipi di scene, ecc.]
Il dataset coco è stato utilizzato principalmente per validare la capacità di distillazione su immagini singole e testare la pipeline in un contesto più semplice. BlendedMVS, invece, è stato scelto per testare la distillazione in un contesto multi-view e per controllare se la consistency loss è matematicamente corretta.
Il dataset BlendedMVS è stato pre-processato utilizzando il wai-processing di MapAnything che si è svolto in 3 step: conversion, covisibility, ed esecuzione del modello MOGE per produrre depth, mask, normals. Anche se per l'obiettivo di questa tesi non è necessario avere tutte queste informazioni, abbiamo deciso di produrle comunque per avere un dataset più completo e versatile per futuri esperimenti, in cui sarà necessario fare il finetuning anche della backbone, e quindi sarà necessario avere a disposizione anche le depth supervision.

4.2 Implementation Details:
- hardware: cluster Euler ETH, NVIDIA RTX 4090 (distributed training)
- framework: PyTorch
- iperparametri

per la questione iperparametri: sono stati identificati ed utilizzati iperparametri di base, che vengono scalati in base al batch size effettivo [breve spiegazione ed esempio di batch size effettivo in contesto multi gpu e multi-view]. Il learning rate è stato scalato in base al batch size effettivo seguendo la regola empirica di linear scaling.
Per identificare il LR di base, sono stati condotti test grid search sul dataset coco per single view e su blendedmvs per multi-view. È stato tenuto conto dei valori maggiormente presenti in letteratura per distillazione e per training di modelli simili, inoltre è stato considerato anche il trade-off tra maggior percentuale di allocazione di memoria gpu e il wall clock time per un'epoca per avere un buon compromesso tra velocità di training e stabilità dell'ottimizzazione. Si è dimostrato che i LR presenti in letteratura sono ottimali per il nostro setting (5e-4 sia SV che MV). [inserire placeholder tabella].
Sono stati testati diversi scheduler (cosine annealing, step, reduceonplateau, no scheduler) e si è dimostrato che lo scheduling del LR non è importante in quanto [vedi dopo] questione del plateau è strutturale e non di ottimizzazione.
weight decay impostato a 1e-4
clip grad 1.0
accum iter 1
automatic mixed precision (AMP) abilitato con bf16
seed fisso a 0 per garantire la riproducibilità
optimizer adamW con betas (0.9, 0.95) e min_lr 1e-6
input size SV variabile in base alla dimensione originale dell'immagine
input size MV fissato a 512x512 per bilanciare qualità, velocità di training e visualizzazione features (PCA)
Per quanto riguarda data augmentation, sono state testate jitter, Gaussian blur,
and grayscale conversion, ma, come per la questione LR scheduler, si è dimostrato che non hanno un impatto significativo sulla convergenza o sulla qualità delle features apprese, probabilmente a causa del plateau strutturale che limita la capacità del modello di apprendere rappresentazioni più complesse.

- overfit su singola view
- test estensivo su tutto il dataset
- identificazione del bottleneck
- dimostrazione grafica con curve di loss (plateau)
- analisi artefatti strutturali del teacher che il student non riesce a replicare
- spiegazione di cosa sono gli artefatti e perché non è importante che il student li replichi
- dimostrazione che il problema è strutturale e non di ottimizzazione
- comparazione curve di loss con diversi iperparametri per dimostrare che il problema è strutturale
- introduzione intuizione strategia di unfreeze
- dimostrazione che l'unfreeze è l'unica strategia che rompe il plateau
- comparazione curve di loss con e senza unfreeze per dimostrare l'efficacia
- ablazione utilizzo solo encoder dino senza MV transformer e completo