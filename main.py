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
    def __init__(self, df):
        self.df = df
        
        self.root = tk.Tk()
        self.root.title("Processamento e Análise de Imagens - Trabalho Prático")
        self.root.geometry("800x450")
        
        self.image_label = tk.Label(self.root)
        self.info_frame = tk.Frame(self.root)
        
        tk.Label(self.info_frame, text="Informações do Paciente", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))

        self.info_vars = {
            "Group": tk.StringVar(value="N/A"),
            "Age": tk.StringVar(value="N/A"),
            "M/F": tk.StringVar(value="N/A"),
            "MMSE": tk.StringVar(value="N/A"),
            "CDR": tk.StringVar(value="N/A")
        }

        for label_text, str_var in self.info_vars.items():
            frame = tk.Frame(self.info_frame)
            frame.pack(anchor="w", pady=2)
            tk.Label(frame, text=f"{label_text}:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
            tk.Label(frame, textvariable=str_var, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)



        self.label = tk.Label(self.root)
        self.label.pack(padx=10, pady=10)

        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        
        file_menu.add_command(label="Abrir Imagem", command=self.open_img)
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        self.root.config(menu=menubar)

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
        
        try:
            patient_data = self.df.loc[mri_id]
            self.info_vars["Group"].set(patient_data.get("Group", "N/A"))
            self.info_vars["Age"].set(patient_data.get("Age", "N/A"))
            self.info_vars["M/F"].set(patient_data.get("M/F", "N/A"))
            self.info_vars["MMSE"].set(patient_data.get("MMSE", "N/A"))
            self.info_vars["CDR"].set(patient_data.get("CDR", "N/A"))
        except KeyError:
            for var in self.info_vars.values():
                var.set("ID não encontrado")
        
        nifti_file = nib.load(filepath)
        slice_rotated = nifti_file.get_fdata()
        
        slice_rotated = np.rot90(slice_rotated) # rotate 90 
        
        if np.max(slice_rotated) != np.min(slice_rotated):
            slice_norm = (slice_rotated - np.min(slice_rotated)) / (np.max(slice_rotated) - np.min(slice_rotated)) * 255
        else:
            slice_norm = np.zeros_like(slice_rotated, dtype=np.uint8)
                
        slice_norm = slice_norm.astype(np.uint8)
        pil_img = Image.fromarray(slice_norm)

        pil_img = pil_img.resize((400, 400), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.label.config(image=tk_img)
    
        self.label.image = tk_img
        
       
    def run(self):
        self.root.mainloop()
        
        
def load_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, sep=';') 
        df.set_index('MRI ID', inplace=True)
        for col in ['nWBV', 'MMSE', 'CDR']:
            if df[col].dtype == 'object': # se a coluna for texto, converte para float
                df[col] = pd.to_numeric(df[col].str.replace(',', '.', regex=False), errors='coerce')
        
        return df
    except FileNotFoundError:
        messagebox.showerror("Error", f"Arquivo {filepath} não encontrado!")
        return None
        
        
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
    
    all_df = load_csv(raw_path)
    
    if is_dataset_split(train_path, val_path, test_path):
        train_df = load_csv(train_path)
        validation_df = load_csv(val_path)
        test_df = load_csv(test_path)
    else:
        train_df, validation_df, test_df = prepare_and_split_data(all_df, output_dir=DATASET_DIR)

    eda = EDA(all_df)
    eda.run()
    
    gui = GUI(all_df)
    gui.run()
