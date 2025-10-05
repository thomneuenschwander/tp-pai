import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from typing import Optional

class EDA:
    def __init__(self, df, out_dir="./statistics"):
        self.df = df
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def already_done(self):
        return os.path.exists(os.path.join(self.out_dir, "stats.json"))

    def run(self):
        if self.already_done():
            return
        self.save_stats()
        self.save_graphs()
        print("EDA executada e persistida em", self.out_dir)

    def save_stats(self):
        stats = {
            "dataset_info": {
                "num_rows": len(self.df),
                "columns": list(self.df.columns)
            },
            "group_counts": self.df["Group"].value_counts().to_dict(),
            "describe": self.df.describe(include="all").fillna("").to_dict()
        }

        with open(os.path.join(self.out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

    def save_graphs(self):        
        # PIE CHART - Groups distr
        group_counts = self.df["Group"].value_counts()
        plt.figure(figsize=(6,6))
        plt.pie(group_counts.values.tolist(), labels=group_counts.index.tolist(), autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        plt.title("Distribuição de Demência na Amostra")
        plt.savefig(os.path.join(self.out_dir, "group_pie.png"))
        plt.close()

        # BOXPLOT - Age X Group
        plt.figure(figsize=(6,4))
        sns.boxplot(x="Group", y="Age", data=self.df)
        plt.title("Idade por Grupo")
        plt.savefig(os.path.join(self.out_dir, "age_boxplot.png"))
        plt.close()

        # BOXPLOT - MMSE X Group
        plt.figure(figsize=(6,4))
        sns.boxplot(x="Group", y="MMSE", data=self.df)
        plt.title("MMSE por Grupo")
        plt.savefig(os.path.join(self.out_dir, "mmse_boxplot.png"))
        plt.close()

        # BOXPLOT - nWBV X Group
        plt.figure(figsize=(6,4))
        sns.boxplot(x="Group", y="nWBV", data=self.df)
        plt.title("nWBV por Grupo")
        plt.savefig(os.path.join(self.out_dir, "nwbv_boxplot.png"))
        plt.close()

        # HISTOGRAM: Age distr
        plt.figure(figsize=(6,4))
        plt.hist(self.df["Age"], bins=20, color="skyblue", edgecolor="black")
        plt.xlabel("Idade")
        plt.ylabel("Frequência")
        plt.title("Distribuição da Idade")
        plt.savefig(os.path.join(self.out_dir, "age_histogram.png"))
        plt.close()


class GUI:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        # BASE WINDOW
        self.root = tk.Tk()
        self.root.title("Processamento e Análise de Imagens - Trabalho Prático")
        self.root.geometry("800x450")
        
        # IMAGE HANDLING
        self.image: Optional[Image.Image] = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 3.0

        # FRAMES
        image_frame = tk.Frame(
            master=self.root,
            bd=2,
        )
        info_frame = tk.Frame(
            master=self.root,
            bd=2,
        )
        image_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        info_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # WIDGETS LEFT FRAME
        self.image_label = tk.Label(master=image_frame, text="Abra uma imagem", font=("Helvetica", 16), bg="#B1B1B1", fg="white")
        self.image_label.pack(fill="both", expand=True)
        
        # WIDGETS RIGHT FRAME
        info_title = tk.Label(
            master=info_frame,
            text="Informações do Paciente",
            font=("Helvetica", 18, "bold")
        )
        info_title.grid(row=0, column=0, columnspan=2, pady=(10, 20))

        self.info = {
            "Group": tk.StringVar(value="N/A"),
            "Age": tk.StringVar(value="N/A"),
            "M/F": tk.StringVar(value="N/A"),
            "MMSE": tk.StringVar(value="N/A"),
            "CDR": tk.StringVar(value="N/A")
        }
    
        for i, (key, value) in enumerate(self.info.items()):
            key_label = tk.Label(master=info_frame, text=f"{key}:", font=("Helvetica", 14, "bold"))
            key_label.grid(row=i+1, column=0, sticky="w", padx=10, pady=2)
            value_label = tk.Label(master=info_frame, textvariable=value, font=("Helvetica", 14))
            value_label.grid(row=i+1, column=1, sticky="w", padx=10, pady=2)

        # MENU
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Abrir Imagem", command=self.open_img)
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        self.root.config(menu=menubar)

        # SCROLL
        self.root.bind("<MouseWheel>", self.on_zoom)     # Windows/MAC
        self.root.bind("<Button-4>", self.on_zoom_linux) # Linux scroll up
        self.root.bind("<Button-5>", self.on_zoom_linux) # Linux scroll down

    def open_img(self):
        initialdir = os.path.join(os.getcwd(), "dataset", "sag")
        filepath = filedialog.askopenfilename(
            initialdir=initialdir, 
            title="Selecione uma imagem",
            filetypes=[("NIfTI", "*.nii.gz")]
        )
        
        if not filepath:
            return
        
        filename = os.path.basename(filepath)
        mri_id = filename.split('_sag')[0]
        
        self.set_info(mri_id)
                
        nifti_file = nib.load(filepath)
        
        data = np.rot90(nifti_file.get_fdata())
        
        self.image = Image.fromarray(data)
        self.display_image()

        
    def set_info(self, mri_id: str):
        try:
            patient_data = self.df.loc[mri_id]
            self.info["Group"].set(patient_data.get("Group", "N/A"))
            self.info["Age"].set(patient_data.get("Age", "N/A"))
            self.info["M/F"].set(patient_data.get("M/F", "N/A"))
            self.info["MMSE"].set(patient_data.get("MMSE", "N/A"))
            self.info["CDR"].set(patient_data.get("CDR", "N/A"))
        except KeyError:
            for var in self.info.values():
                var.set("ID não encontrado")
       
       
    def display_image(self):
        if self.image is None:
            return
        
        w, h = self.image.size
        new_size = (int(w * self.zoom_factor), int(h * self.zoom_factor))
        
        resized_img = self.image.resize(new_size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized_img)

        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img
        
    def on_zoom(self, event):
        """Detecta zoom com scroll do mouse no Windows/MAC"""
        if event.delta > 0:
            self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
        else:
            self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)
        self.display_image()

    def on_zoom_linux(self, event):
        """Detecta zoom com scroll do mouse no Linux"""
        if event.num == 4:  # scroll up
            self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
        elif event.num == 5:  # scroll down
            self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)
        self.display_image()
    
    def run(self):
        self.root.mainloop()
   
   
        
def prepare_and_split_data(df: pd.DataFrame, output_dir: str):
    """
    - Converted -> Demented: se CDR <= 0
    - Converted -> Nondemented: se CDR > 0
    - train subset: 80% dos pacientes com todos os exames das suas visitas.
    - test subset: 20% dos pacientes com todos os exames das suas visitas.
    - validation subset: 20% do conjunto de treino
    """
    
    df_copy = df.copy()
    df_copy['Group'] = df_copy.apply(lambda row: 'Demented' if row['Group'] == 'Converted' and row['CDR'] > 0 else 'Nondemented' if row['Group'] == 'Converted' and row['CDR'] == 0 else row['Group'], axis=1)
    df_copy = df_copy[df_copy['Group'].isin(['Demented', 'Nondemented'])]

    patient_ids = df_copy['Subject ID'].unique()
    patient_labels = df_copy.groupby('Subject ID')['Group'].last()

    split_seed = 42

    train_val_patients, test_patients = train_test_split(
        patient_labels.index,
        test_size=0.2,
        stratify=patient_labels.values,
        random_state=split_seed
    )

    train_patients, val_patients, _, _ = train_test_split(
        train_val_patients,
        patient_labels[train_val_patients].values,
        test_size=0.2,
        stratify=patient_labels[train_val_patients].values,
        random_state=split_seed
    )
    
    train_set = df_copy[df_copy['Subject ID'].isin(train_patients)]
    validation_set = df_copy[df_copy['Subject ID'].isin(val_patients)]
    test_set = df_copy[df_copy['Subject ID'].isin(test_patients)]

    os.makedirs(output_dir, exist_ok=True)
    train_set.to_csv(os.path.join(output_dir, 'train_set.csv'))
    validation_set.to_csv(os.path.join(output_dir, 'validation_set.csv'))
    test_set.to_csv(os.path.join(output_dir, 'test_set.csv'))

    return train_set, validation_set, test_set

def is_dataset_split(train_path: str, val_path: str, test_path: str) -> bool:
    return os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)

if __name__ == "__main__":
    DATASET_DIR = './dataset'
    
    raw_path = os.path.join(DATASET_DIR, 'oasis_longitudinal_demographic.csv')
    train_path = os.path.join(DATASET_DIR, 'train_set.csv')
    val_path = os.path.join(DATASET_DIR, 'validation_set.csv')
    test_path = os.path.join(DATASET_DIR, 'test_set.csv')
    
    all_df = df = pd.read_csv(raw_path, sep=';', decimal=',', index_col='MRI ID')
    
    
    if is_dataset_split(train_path, val_path, test_path):
        train_df = pd.read_csv(train_path, index_col='MRI ID')
        validation_df = pd.read_csv(val_path, index_col='MRI ID')
        test_df = pd.read_csv(test_path, index_col='MRI ID')
    else:
        train_df, validation_df, test_df = prepare_and_split_data(all_df, output_dir=DATASET_DIR)

    eda = EDA(all_df)
    eda.run()
    
    gui = GUI(all_df)
    gui.run()
