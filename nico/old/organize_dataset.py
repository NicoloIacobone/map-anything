import os
import shutil
from pathlib import Path

# Modifica questo path con la tua cartella dataset
DATASET_PATH = "/Users/nicoloiacobone/Desktop/nico/UNIVERSITA/MAGISTRALE/Tesi/Tommasi/Zurigo/git_clones/distillation/dataset/train"
OUTPUT_PATH = "/Users/nicoloiacobone/Desktop/nico/UNIVERSITA/MAGISTRALE/Tesi/Tommasi/Zurigo/git_clones/distillation/dataset/ETH3D_single/images/train"

def organize_dataset(dataset_path, output_path):
    """
    Raggruppa tutte le immagini PNG da tutte le sottocartelle
    e le rinomina sequenzialmente da 00000 a NNNNN
    """
    # Crea la cartella di output se non esiste
    os.makedirs(output_path, exist_ok=True)
    
    # Raccogli tutti i file PNG
    image_counter = 0
    dataset_root = Path(dataset_path)
    
    # Itera su tutte le sottocartelle (scene)
    for scene_folder in sorted(dataset_root.iterdir()):
        if scene_folder.is_dir():
            # Trova tutti i PNG nella cartella della scena
            for image_file in sorted(scene_folder.glob("*.png")):
                # Crea il nuovo nome
                new_name = f"{image_counter:05d}.png"
                new_path = os.path.join(output_path, new_name)
                
                # Copia il file
                shutil.copy2(image_file, new_path)
                print(f"Copied: {image_file.name} -> {new_name}")
                
                image_counter += 1
    
    print(f"\nTotal images processed: {image_counter}")

if __name__ == "__main__":
    organize_dataset(DATASET_PATH, OUTPUT_PATH)