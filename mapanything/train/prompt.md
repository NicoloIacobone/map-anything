Crea un notebook Jupyter completo per il debug visuale della **Multi-View Consistency Loss** 
della classe SemanticConsistencyLoss in mapanything/train/losses.py.

## OBIETTIVO
Visualizzare step-by-step il calcolo della consistency loss per capire se:
1. La proiezione geometrica dei punti 3D è corretta
2. L'occlusion check funziona come previsto
3. Le feature semantiche estratte sono semanticamente simili
4. Quanti pixel trovano corrispondenze valide
5. Dove fallisce il matching (se fallisce)

## INPUT RICHIESTI
Il notebook deve accettare (tramite celle iniziali configurabili):
- `batch_idx`: indice del batch su cui fare debug (default: 0)
- `view_i`: indice vista anchor (default: 0)
- `view_j`: indice vista target (default: 1)
- `grid_step`: spaziatura griglia punti debug (default: 8, per visualizzare ogni 8x8 pixel)
- `checkpoint_path`: path al checkpoint con batch e predizioni salvate
- `device`: 'cuda' o 'cpu'

## STRUTTURA DEL NOTEBOOK

### Sezione 1: SETUP E LOADING
- Import necessari (torch, matplotlib, numpy, sklearn.decomposition.PCA, cv2)
- Caricamento del batch e delle predizioni da checkpoint
- Estrazione di batch[view_i], batch[view_j], preds[view_i], preds[view_j]

### Sezione 2: PREPARAZIONE DATI (replica STEP 1-4 della loss)
Mostrare:
- Forma e statistiche di:
- anchor_pts_world: (B, H_orig, W_orig, 3)
- anchor_feat: (B, C, H_orig, W_orig) - upsampled a risoluzione originale
- anchor_conf: (B, 1, H_orig, W_orig)
- tgt_feat_upsampled: (B, C, H_orig, W_orig)
- tgt_conf: (B, 1, H_orig, W_orig)

### Sezione 3: VISUALIZZAZIONE PER-PIXEL PCA
Per la griglia di punti selezionati (ogni grid_step pixel):
- **Subplot A**: Immagine anchor (vista i) con pallini overlay ai punti selezionati
- **Subplot B**: Feature map anchor convertita con PCA a 3 canali RGB
  - Istruzioni: applicare PCA alle features (B, C, H, W) → (B, 3, H, W)
  - Normalizzare in [0, 255] per visualizzazione
- **Subplot C**: Per ogni punto, mostrare il colore PCA nel punto anchor

### Sezione 4: PROIEZIONE (STEP 6)
Replicate `project_to_view()`:
- Mostrare grid output: coordinate normalizzate [-1, 1]
- Mostrare z_proj: profondità proiettata
- Mostrare fov_mask: quali punti rimangono dentro [-1, 1]
- Contare: N punti totali vs N punti in FOV

Visualizzazione:
- **Subplot A**: Immagine target (vista j) con proiezioni overlay
- **Subplot B**: Heatmap FOV mask (rosso=dentro, blu=fuori)
- **Subplot C**: Heatmap profondità z_proj normalizzata

### Sezione 5: CAMPIONAMENTO (STEP 7)
Replicate `F.grid_sample()`:
- sampled_feat: feature estratte sulla vista j
- sampled_conf: confidenze estratte sulla vista j

Visualizzazione:
- **Subplot A**: PCA delle feature campionate (tgt_feat_upsampled nel grid)
- **Subplot B**: Heatmap confidenze campionate
- **Subplot C**: Per punti selezionati, mostrare colore PCA sampato vs anchor

### Sezione 6: OCCLUSION CHECK (STEP 8)
- tgt_depth_map: profondità GT della vista j campionata nel grid
- z_proj vs sampled_tgt_depth: differenze
- occ_mask: torch.abs(z_proj - sampled_tgt_depth) < self.occ_thresh

Visualizzazione:
- **Subplot A**: Heatmap |z_proj - sampled_tgt_depth|
- **Subplot B**: Heatmap occ_mask (verde=visibile, rosso=occluso)
- **Subplot C**: Combine fov_mask + occ_mask = valid_mask finale
- Statistiche: % punti FOV che passano occlusion check

### Sezione 7: ACCUMULO (STEP 9)
- valid_mask finale
- has_matches_mask: tracking quale pixel ha almeno 1 match valido
- sum_weighted_feat: accumulo ponderato

Visualizzazione:
- **Subplot A**: Heatmap valid_mask per questa coppia (vista i, j)
- **Subplot B**: Heatmap has_matches_mask (accumulo di tutte le viste j≠i)
- **Subplot C**: % di pixel con ≥1 match valido

### Sezione 8: MEDIA PONDERATA (STEP 10)
- mean_feat: media ponderata delle feature campionate
- Normalizzazione L2

Visualizzazione:
- **Subplot A**: PCA della mean_feat
- **Subplot B**: Per punti selezionati, confronto:
  - Color anchor_feat (rosso)
  - Color mean_feat (blu)
  - Sovrapposizione per vedere similarità

### Sezione 9: LOSS FINALE (STEP 11)
- sim_i: cosine similarity tra anchor_feat e mean_feat (normalizzate)
- loss_i: 1.0 - sim_i

Visualizzazione:
- **Subplot A**: Heatmap cosine similarity (1.0=identici, -1.0=opposti)
- **Subplot B**: Heatmap loss per-pixel
- **Subplot C**: Distribution della loss (istogramma)
- Statistiche: media loss, min, max, # pixel con loss > threshold

### Sezione 10: SUMMARY LOOP
**Stesso per tutte le coppie (i, j)**
- Loop su view_i in range(n_views)
- Per ogni view_i, loop su view_j in range(n_views), j ≠ i
- Mostrare small multiples (griglia di plot) con tutte le coppie
- Tabella riepilogativa: view_i, view_j, % match, avg_loss

## FUNZIONI HELPER DA IMPLEMENTARE

```python
def apply_pca_to_features(feat, n_components=3):
    """
    Input: (B, C, H, W)
    Output: (B, 3, H, W) in [0, 255]
    """
    # Reshape feat a (B*H*W, C)
    # Applicare PCA fit+transform
    # Reshape back e normalizzare
    
def draw_points_on_image(img, coords_2d, colors=None, radius=5):
    """
    Disegna punti 2D sopra immagine
    coords_2d: (N, 2) in pixel
    colors: (N, 3) RGB o None per default
    """
    
def compute_cosine_similarity(feat1, feat2):
    """
    Input: (B, C, H, W) each
    Output: (B, H, W) similarity in [-1, 1]
    """
    # Normalizzare feat1 e feat2 su dimensione C
    # Dot product
```

## DATI NECESSARI
Deve essere salvato un checkpoint contenente:
{
    'batch': batch,           # lista di dizionari (n_views)
    'preds': preds,          # lista di dizionari (n_views)
}

Con chiavi obbligatorie:
- batch[i]: 'pts3d', 'pts3d_cam', 'semantics', 'camera_pose', 'camera_intrinsics'
- preds[i]: 'semantics', 'sem_conf', 'pts3d'

## PARAMETRI CONFIGURABILI (celle setup)
BATCH_IDX = 0
VIEW_I = 0
VIEW_J = 1
GRID_STEP = 8
CHECKPOINT_PATH = '/path/to/debug_checkpoint.pt'
DEVICE = 'cuda'
PCA_COMPONENTS = 3
OCCLUSION_THRESHOLD = 0.1
MIN_CONF = 1.0

OUTPUT ATTESO
Il notebook deve permettere di:

✅ Visualizzare esattamente dove vanno i punti 3D quando proiettati
✅ Capire quanti pixel trovano match (e quanti no, e perché)
✅ Vedere se le feature semantiche matched sono effettivamente simili
✅ Identificare bottleneck nella consistency loss (FOV? occlusion? feature?)
✅ Confrontare visivamente anchor_feat vs mean_feat per capire se la loss ha senso