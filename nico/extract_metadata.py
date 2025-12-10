import h5py
import os
import numpy as np
from tqdm import tqdm

# ================= CONFIGURAZIONE =================
# Percorso del file HDF5 sorgente
<<<<<<< HEAD
H5_PATH = "/Users/nicoloiacobone/Desktop/nico/UNIVERSITA/MAGISTRALE/Tesi/Tommasi/Zurigo/git_clones/distillation/dataset/DL3DV.hdf5"

# Dove estrarre i dati (crea questa cartella se non esiste)
EXTRACT_ROOT = "/Users/nicoloiacobone/Desktop/nico/UNIVERSITA/MAGISTRALE/Tesi/Tommasi/Zurigo/git_clones/distillation/dataset/DL3DV_Extracted"
=======
H5_PATH = "/scratch2/nico/distillation/dataset/DL3DV.hdf5"

# Dove estrarre i dati (crea questa cartella se non esiste)
EXTRACT_ROOT = "/scratch2/nico/distillation/dataset/DL3DV_Extracted"
>>>>>>> 353dfdd (debug - added some utils code)
# ==================================================

def save_data(name, dataset, output_root):
    """
    Salva un dataset HDF5 su disco.
    Gestisce automaticamente immagini (binarie) e testo.
    """
    # Costruisci il path finale
    # name è tipo "1K/0000/00001.jpg"
    final_path = os.path.join(output_root, name)
    
    # Assicurati che la cartella padre esista
    parent_dir = os.path.dirname(final_path)
    os.makedirs(parent_dir, exist_ok=True)
    
    try:
        # Leggi i dati grezzi dal dataset
        # [()] legge tutto il contenuto in memoria
        data = dataset[()]
        
        # --- CASO 1: IMMAGINI (.jpg, .png) ---
        if name.lower().endswith(('.jpg', '.png', '.jpeg')):
            with open(final_path, 'wb') as f:
                # Se è un oggetto numpy void/bytes (blob binario)
                if isinstance(data, (np.void, bytes, np.bytes_)):
                    f.write(data.tobytes() if hasattr(data, 'tobytes') else data)
                # Se è un array numpy di uint8 (comune in h5)
                elif isinstance(data, np.ndarray):
                    f.write(data.tobytes())
                else:
                    print(f"[WARN] Tipo dati inatteso per immagine {name}: {type(data)}")

        # --- CASO 2: TESTO (.json, .txt) ---
        elif name.lower().endswith(('.json', '.txt', '.csv')):
            # Decodifica stringa
            text_content = ""
            if isinstance(data, (bytes, np.bytes_)):
                text_content = data.decode('utf-8')
            elif isinstance(data, np.ndarray):
                if data.size == 1:
                    val = data.item()
                    text_content = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                else:
                    # Array di righe
                    lines = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in data]
                    text_content = "\n".join(lines)
            
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
        
        # --- CASO 3: ALTRO (Matrici pose, calibration, ecc) ---
        # Se trovi file .npy o senza estensione, li salviamo come testo o npy
        else:
            # Fallback generico: prova a salvare come binario se sembra binario, o testo
            with open(final_path, 'wb') as f:
                if hasattr(data, 'tobytes'):
                    f.write(data.tobytes())
                else:
                    f.write(data)

    except Exception as e:
        print(f"\n[ERR] Errore salvando {name}: {e}")

def main():
    if not os.path.exists(H5_PATH):
        print(f"File non trovato: {H5_PATH}")
        return

    print(f"--- Estrazione Totale da {H5_PATH} a {EXTRACT_ROOT} ---")
    
    with h5py.File(H5_PATH, 'r') as f:
        # 1. Contiamo gli oggetti per la progress bar (opzionale ma consigliato per 75GB)
        print("Calcolo numero totale di file (potrebbe richiedere un minuto)...")
        total_items = 0
        def count_items(name, obj):
            nonlocal total_items
            if isinstance(obj, h5py.Dataset):
                total_items += 1
        f.visititems(count_items)
        print(f"Totale file da estrarre: {total_items}")
        
        # 2. Estrazione vera e propria
        with tqdm(total=total_items, unit="file") as pbar:
            def extract_visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    save_data(name, obj, EXTRACT_ROOT)
                    pbar.update(1)
                elif isinstance(obj, h5py.Group):
                    # Crea la cartella anche se è vuota (per struttura)
                    target_dir = os.path.join(EXTRACT_ROOT, name)
                    os.makedirs(target_dir, exist_ok=True)
            
            f.visititems(extract_visitor)

    print("\nEstrazione completata!")

if __name__ == "__main__":
    main()