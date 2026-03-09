import os
import shutil

# --- CONFIGURAZIONE PERCORSI ---
input_dir = "/scratch2/nico/distillation/dataset/scannet/scans/scene0000_00/sens/color"
output_dir = "/scratch2/nico/distillation/dataset/scannet/scans/scene0000_00/sens/subsampling"

# Fattore di campionamento: 1 frame ogni K
# ScanNet ha circa 30 fps. K=30 significa prendere 1 frame al secondo.
SUBSAMPLE_FACTOR = 60

def main():
    # 1. Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # 2. Leggi tutti i file .jpg o .png dalla cartella input
    # (Filtriamo per estensione per evitare di leggere file nascosti)
    valid_extensions = ('.jpg', '.png', '.jpeg')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"Nessuna immagine trovata in {input_dir}")
        return

    # 3. Ordinamento numerico (Naturale)
    # Assumiamo che i file si chiamino tipo "0.jpg", "1.jpg", "123.jpg"
    # Estraiamo il numero prima del punto per ordinarli matematicamente
    files.sort(key=lambda x: int(x.split('.')[0]))

    # 4. Applica il subsampling (prendi 1 ogni SUBSAMPLE_FACTOR)
    subsampled_files = files[::SUBSAMPLE_FACTOR]

    print(f"Trovate {len(files)} immagini totali.")
    print(f"Verranno copiate {len(subsampled_files)} immagini (1 ogni {SUBSAMPLE_FACTOR})...")

    # 5. Copia i file selezionati nella cartella di destinazione
    for file_name in subsampled_files:
        src_path = os.path.join(input_dir, file_name)
        dst_path = os.path.join(output_dir, file_name)
        
        shutil.copy(src_path, dst_path)

    print(f"Copia completata! Le immagini subsamplate sono in:\n{output_dir}")

if __name__ == "__main__":
    main()