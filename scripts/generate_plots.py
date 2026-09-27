import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- CONFIGURAZIONI ---
BASE_DIR = "/scratch2/nico/distillation/output/embeddings"
TEACHER_DIR = os.path.join(BASE_DIR, "teacher_embeddings")
STUDENT_DIR = os.path.join(BASE_DIR, "student_embeddings")
OUTPUT_DIR = "/scratch2/nico/distillation/output/feature_plots"

SCENARIOS = ['1', '10', '100', '1000']
EPOCHS = [0, 500, 1000, 1500]
IMAGE_SUFFIXES = [49, 136, 142, 149, 154, 260, 283, 397, 443, 446]

# Creiamo la cartella di output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_coco_id(suffix):
    """Formatta il suffisso numerico in un ID COCO a 12 cifre."""
    return f"{suffix:012d}"

def load_and_flatten_tensor(path):
    """Carica il tensore, lo sposta su CPU e lo appiattisce a (256, N_pixels)."""
    if not os.path.exists(path):
        return None
    tensor = torch.load(path, map_location='cpu', weights_only=True)
    # Assumiamo che il tensore sia (B, C, H, W) o (C, H, W) con C=256
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0) # Rimuove il batch size se presente -> (C, H, W)
    
    C = tensor.shape[0]
    return tensor.view(C, -1).numpy() # -> (256, H*W)

def generate_plots(scenario, epoch, teacher_feats, student_feats):
    """Genera e salva i 3 tipi di grafici."""
    print(f"Generazione grafici per Scenario: {scenario}, Epoca: {epoch}...")
    
    # Prepariamo la cartella specifica
    save_dir = os.path.join(OUTPUT_DIR, f"scenario_{scenario}_epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. KDE PLOT (Distribuzione Globale)
    # Campioniamo 50.000 valori casuali per non far esplodere la RAM
    np.random.seed(42)
    t_flat = np.random.choice(teacher_feats.flatten(), size=50000, replace=False)
    s_flat = np.random.choice(student_feats.flatten(), size=50000, replace=False)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(t_flat, fill=True, label="Teacher (SAM2)", color="royalblue", alpha=0.5)
    sns.kdeplot(s_flat, fill=True, label="Student", color="darkorange", alpha=0.5)
    plt.title(f"Global Feature Distribution (KDE) - Overfit {scenario} imgs, Epoch {epoch}")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "1_kde_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. VIOLIN PLOT (Analisi Canale per Canale)
    # Scegliamo 5 canali rappresentativi a caso (es. 10, 50, 100, 150, 200)
    channels_to_plot = [10, 50, 100, 150, 200]
    plot_data = []
    
    for ch in channels_to_plot:
        # Campioniamo 1000 pixel per canale per il violin
        t_ch = np.random.choice(teacher_feats[ch, :], size=1000, replace=False)
        s_ch = np.random.choice(student_feats[ch, :], size=1000, replace=False)
        
        for val in t_ch:
            plot_data.append({"Channel": f"Ch {ch}", "Model": "Teacher", "Value": val})
        for val in s_ch:
            plot_data.append({"Channel": f"Ch {ch}", "Model": "Student", "Value": val})
            
    df_violin = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_violin, x="Channel", y="Value", hue="Model", split=True, inner="quart", palette={"Teacher": "royalblue", "Student": "darkorange"})
    plt.title(f"Per-Channel Distribution (Violin Plot) - Overfit {scenario} imgs, Epoch {epoch}")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "2_violin_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. t-SNE SCATTER PLOT (Analisi Spazio Latente)
    # Attenzione: t-SNE è molto pesante. Campioniamo solo 1000 "pixel" (vettori 256D) per modello
    n_samples = 1000
    t_idx = np.random.choice(teacher_feats.shape[1], n_samples, replace=False)
    s_idx = np.random.choice(student_feats.shape[1], n_samples, replace=False)
    
    # Estraiamo i vettori (forma: [N_samples, 256])
    t_sample = teacher_feats[:, t_idx].T
    s_sample = student_feats[:, s_idx].T
    
    # Uniamo per il t-SNE
    combined_features = np.vstack((t_sample, s_sample))
    labels = ["Teacher"] * n_samples + ["Student"] * n_samples
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(combined_features)
    
    df_tsne = pd.DataFrame({
        "t-SNE Dim 1": tsne_results[:, 0],
        "t-SNE Dim 2": tsne_results[:, 1],
        "Model": labels
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x="t-SNE Dim 1", y="t-SNE Dim 2", hue="Model", palette={"Teacher": "royalblue", "Student": "darkorange"}, alpha=0.6, s=15)
    plt.title(f"t-SNE Latent Space Projection - Overfit {scenario} imgs, Epoch {epoch}")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "3_tsne_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- ESECUZIONE PRINCIPALE ---
for scenario in SCENARIOS:
    # Se lo scenario è "1", usiamo solo l'immagine 49, altrimenti tutte
    current_images = [49] if scenario == '1' else IMAGE_SUFFIXES
    
    for epoch in EPOCHS:
        all_teacher_feats = []
        all_student_feats = []
        
        for img_suffix in current_images:
            coco_id = get_coco_id(img_suffix)
            
            # Il teacher ha solo epoch 0
            teacher_path = os.path.join(TEACHER_DIR, f"{coco_id}_epoch_0.pt")
            student_path = os.path.join(STUDENT_DIR, scenario, f"{coco_id}_epoch_{epoch}.pt")
            
            t_feat = load_and_flatten_tensor(teacher_path)
            s_feat = load_and_flatten_tensor(student_path)
            
            if t_feat is not None and s_feat is not None:
                all_teacher_feats.append(t_feat)
                all_student_feats.append(s_feat)
            else:
                print(f"Warning: File mancanti per immagine {coco_id} (Scenario {scenario}, Epoca {epoch})")
        
        if all_teacher_feats and all_student_feats:
            # Concateniamo le feature lungo l'asse dei pixel -> (256, N_tot_pixels)
            agg_teacher = np.concatenate(all_teacher_feats, axis=1)
            agg_student = np.concatenate(all_student_feats, axis=1)
            
            generate_plots(scenario, epoch, agg_teacher, agg_student)
        else:
            print(f"Nessun dato trovato per Scenario {scenario}, Epoca {epoch}.")

print("Elaborazione completata! Controlla la cartella 'feature_plots'.")