import os
import time
import requests
import mdtraj as md
import numpy as np
import pandas as pd

# 1. Configuración de archivos
PISCES_FILE = "cullpdb_clean.txt" # Nombre del archivo descargado de PISCES
OUTPUT_CSV = "dataset_aminoacidos_dssp.csv"
PDB_DIR = "pdb_files"

os.makedirs(PDB_DIR, exist_ok=True)

# 2. Leer los PDB IDs y Cadenas del archivo de PISCES
pdb_chain_list = []
with open(PISCES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        # Ignorar comentarios y encabezados
        if not line or line.startswith("#") or line.startswith("IDs"):
            continue
        
        parts = line.split()
        if len(parts) > 0:
            entry = parts[0]  # Ejemplo: '1A8OA'
            if len(entry) >= 5:
                pdb_id = entry[:4].lower()
                chain = entry[4].upper()
                pdb_chain_list.append((pdb_id, chain))

print(f"Total de cadenas a procesar: {len(pdb_chain_list)}")

# 3. Función para obtener el archivo PDB (con caché local)
def get_pdb_file(pdb_id):
    filepath = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(filepath):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            with open(filepath, 'w') as f:
                f.write(response.text)
        else:
            raise Exception(f"Error HTTP {response.status_code} al descargar {pdb_id}")
        time.sleep(0.1) # Pausa leve para cuidar el servidor
    return filepath

# 4. Función para calcular ángulos y DSSP de una cadena específica
def get_angle_map(traj, compute_func):
    """Calcula un ángulo diédrico específico y mapea su valor por residuo en grados."""
    atom_indices, angles = compute_func(traj)
    if len(atom_indices) == 0:
        return {}
    
    angles_deg = np.rad2deg(angles[0])
    top = traj.topology
    
    # En MDTraj, el segundo átomo (índice 1) del cuarteto siempre pertenece al residuo actual
    return {
        top.atom(atoms[1]).residue.index: val 
        for atoms, val in zip(atom_indices, angles_deg)
    }

def process_protein_chain(pdb_id, target_chain):
    filepath = get_pdb_file(pdb_id)
    traj = md.load(filepath)
    top = traj.topology

    # 1. Ángulos Backbone
    phi_map = get_angle_map(traj, md.compute_phi)
    psi_map = get_angle_map(traj, md.compute_psi)
    omega_map = get_angle_map(traj, md.compute_omega)

    # 2. Ángulos Cadena Lateral (Chi 1 a 4)
    chi1_map = get_angle_map(traj, md.compute_chi1)
    chi2_map = get_angle_map(traj, md.compute_chi2)
    chi3_map = get_angle_map(traj, md.compute_chi3)
    chi4_map = get_angle_map(traj, md.compute_chi4)

    # 3. Clasificación DSSP (8 estados)
    dssp = md.compute_dssp(traj, simplified=False)[0]

    data = []
    for res in top.residues:
        if res.is_protein and res.chain.chain_id.upper() == target_chain:
            idx = res.index
            data.append({
                'pdb_id': pdb_id,
                'chain': target_chain,
                'res_seq': res.resSeq,
                'res_name': res.name,
                'dssp': dssp[idx],
                # Backbone
                'phi': phi_map.get(idx, np.nan),
                'psi': psi_map.get(idx, np.nan),
                'omega': omega_map.get(idx, np.nan),
                # Side-chain
                'chi1': chi1_map.get(idx, np.nan),
                'chi2': chi2_map.get(idx, np.nan),
                'chi3': chi3_map.get(idx, np.nan),
                'chi4': chi4_map.get(idx, np.nan)
            })
            
    return data

# 5. Control de reanudación (si ya existen datos guardados)
processed_set = set()
if os.path.exists(OUTPUT_CSV):
    try:
        df_exist = pd.read_csv(OUTPUT_CSV, usecols=['pdb_id', 'chain'])
        for _, row in df_exist.iterrows():
            processed_set.add((str(row['pdb_id']).lower(), str(row['chain']).upper()))
        print(f"Cadenas previamente procesadas y omitidas: {len(processed_set)}")
    except Exception:
        pass

# 6. Bucle principal con guardado en lotes (cada 25 cadenas)
batch_data = []
processed_count = 0

for i, (pdb_id, chain) in enumerate(pdb_chain_list, 1):
    if (pdb_id, chain) in processed_set:
        continue

    try:
        res_data = process_protein_chain(pdb_id, chain)
        if res_data:
            batch_data.extend(res_data)
            processed_count += 1

        # Guardar en el archivo CSV cada 25 cadenas
        if processed_count > 0 and processed_count % 25 == 0:
            df_batch = pd.DataFrame(batch_data)
            file_exists = os.path.exists(OUTPUT_CSV)
            df_batch.to_csv(OUTPUT_CSV, mode='a', index=False, header=not file_exists)
            print(f"[{i}/{len(pdb_chain_list)}] Guardado lote de {processed_count} cadenas en '{OUTPUT_CSV}'")
            batch_data = [] # Limpiar lote
            
    except Exception as e:
        print(f"[{i}/{len(pdb_chain_list)}] Error en {pdb_id}_{chain}: {e}")

# Guardar el último lote pendiente
if batch_data:
    df_batch = pd.DataFrame(batch_data)
    file_exists = os.path.exists(OUTPUT_CSV)
    df_batch.to_csv(OUTPUT_CSV, mode='a', index=False, header=not file_exists)
    print("¡Procesamiento finalizado con éxito!")













