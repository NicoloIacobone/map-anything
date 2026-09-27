import h5py
import sys
import os

# CONFIGURAZIONE
# Cambia questo percorso con il file che vuoi esplorare
FILE_PATH = "/scratch2/nico/distillation/dataset/DL3DV.hdf5"

def format_item(name, obj):
    """Restituisce una stringa formattata per ls"""
    if isinstance(obj, h5py.Group):
        return f"\033[94m{name}/\033[0m"  # Blue per gruppi
    elif isinstance(obj, h5py.Dataset):
        return f"{name} \033[90m({obj.shape}, {obj.dtype})\033[0m" # Grigio per info
    return name

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Errore: File non trovato: {FILE_PATH}")
        return

    try:
        f = h5py.File(FILE_PATH, 'r')
    except Exception as e:
        print(f"Impossibile aprire il file: {e}")
        return

    current_path = "/"
    
    print(f"--- HDF5 Shell: {FILE_PATH} ---")
    print("Comandi disponibili: ls, cd <dir>, cat <dataset>, pwd, exit")

    while True:
        try:
            # Prompt stile shell
            cmd_input = input(f"\033[92mh5shell:{current_path}$ \033[0m").strip().split()
            
            if not cmd_input:
                continue
                
            cmd = cmd_input[0]
            args = cmd_input[1:]

            if cmd == "exit" or cmd == "quit":
                break
                
            elif cmd == "pwd":
                print(current_path)

            elif cmd == "ls":
                # Ottieni l'oggetto corrente
                try:
                    group = f[current_path]
                    keys = list(group.keys())
                    
                    # Se ci sono troppi elementi, mostriamo una sintesi
                    if len(keys) > 50:
                        print(f"Contiene {len(keys)} elementi. Mostro i primi 20 e gli ultimi 5:")
                        preview_keys = keys[:20] + ["..."] + keys[-5:]
                    else:
                        preview_keys = keys

                    for k in preview_keys:
                        if k == "...":
                            print("  ...")
                            continue
                        
                        item_path = f"{current_path}/{k}" if current_path != "/" else k
                        print(f"  {format_item(k, f[item_path])}")
                        
                except Exception as e:
                    print(f"Errore ls: {e}")

            elif cmd == "cd":
                if not args:
                    current_path = "/"
                    continue
                
                target = args[0]
                
                if target == "..":
                    # Torna su
                    if current_path != "/":
                        current_path = os.path.dirname(current_path)
                        if current_path == "": current_path = "/"
                elif target == "/":
                    current_path = "/"
                else:
                    # Costruisci nuovo path
                    new_path = f"{current_path}/{target}" if current_path != "/" else f"/{target}"
                    # Pulisci path (rimuovi doppi slash ecc se capita)
                    new_path = new_path.replace("//", "/")
                    
                    if new_path in f:
                        if isinstance(f[new_path], h5py.Group):
                            current_path = new_path
                        else:
                            print(f"Errore: '{target}' non è una cartella (Gruppo)")
                    else:
                        print(f"Errore: Cartella '{target}' non trovata")

            elif cmd == "cat" or cmd == "info":
                if not args:
                    print("Uso: cat <nome_dataset>")
                    continue
                
                target = args[0]
                target_path = f"{current_path}/{target}" if current_path != "/" else f"/{target}"
                
                if target_path in f:
                    obj = f[target_path]
                    if isinstance(obj, h5py.Dataset):
                        print(f"Nome: {obj.name}")
                        print(f"Shape: {obj.shape}")
                        print(f"Tipo: {obj.dtype}")
                        print(f"Attributi: {list(obj.attrs.keys())}")
                        if obj.size < 10:
                            print(f"Dati: {obj[:]}")
                    else:
                        print(f"'{target}' è un Gruppo, usa 'ls' o 'cd'")
                else:
                    print(f"Dataset '{target}' non trovato")

            else:
                print("Comando sconosciuto. Usa: ls, cd, cat, pwd, exit")

        except KeyboardInterrupt:
            print("\nUsa 'exit' per uscire.")
        except Exception as e:
            print(f"Errore inatteso: {e}")

    f.close()

if __name__ == "__main__":
    main()