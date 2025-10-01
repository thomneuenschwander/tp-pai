import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

FILES = {
    'CORONAL': './dataset/cor/OAS2_0001_MR1_cor.nii.gz',
    'SAGITAL': './dataset/sag/OAS2_0001_MR1_sag.nii.gz',
    'AXIAL': './dataset/axl/OAS2_0001_MR1_axl.nii.gz' 
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 
fig.suptitle("Visualização Coronal, Sagital, Axial", fontsize=16)

for i, (title, path) in enumerate(FILES.items()):
    try:
        img = nib.load(path) 
        data = img.get_fdata() 
    
        if data.ndim == 2:
            plot_data = data
        elif data.ndim == 3:
            slice_index = data.shape[2] // 2 
            plot_data = data[:, :, slice_index]
        else:
            print(f"Aviso: O arquivo {title} tem dimensão inesperada ({data.ndim}).")
            continue
        
        ax = axes[i] 
        
        ax.imshow(plot_data, cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis('off') 

        print(f"SUCESSO: {title} carregada e plotada. Shape: {data.shape}")

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado no caminho: {path}. Pulando {title}.")
    except Exception as e:
        print(f"ERRO ao processar '{title}': {e}.")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()