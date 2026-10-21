"""
Entrenamiento y Evaluacion de Modelos RNA: MLP, CNN 1 Capa, CNN 2 Capas
Trabajo Practico N° 2 - Sistemas Inteligentes - Grupo 9
UTN FRBA - 2026
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "03_Datos")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "05_Resultados_Graficos")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "resultados_entrenamientos")
LOCAL_DATASET_DIR = r"C:\datasets\mushrooms_10"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset personalizado para cargar imagenes locales
# -------------------------------------------------------------
class MushroomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # Pre-cargar rutas y labels para maxima velocidad
        self.samples = []
        for _, row in self.df.iterrows():
            rel_path = row["filename"].replace("/", os.sep)
            full_path = os.path.join(self.root_dir, rel_path)
            label = int(row["clase_id"])
            self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_path, label = self.samples[idx]
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (128, 128), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

# -------------------------------------------------------------
# 2. Definicion de Arquitecturas
# -------------------------------------------------------------

# Modelo 1: Perceptrón Multicapa (MLP) con Backpropagation
class MushroomMLP(nn.Module):
    def __init__(self, input_size=64*64*3, hidden1=256, hidden2=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

# Modelo 2: CNN con 1 Capa Convolucional
class MushroomCNN1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Entrada: 3 x 128 x 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # -> 32 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)         # -> 32 x 32 x 32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Modelo 3: CNN con 2 Capas Convolucionales (filtros aumentan con la profundidad: 32 -> 64)
class MushroomCNN2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Entrada: 3 x 128 x 128
        self.features = nn.Sequential(
            # Capa Conv 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # -> 32 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # -> 32 x 64 x 64
            
            # Capa Conv 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # -> 64 x 32 x 32
            
            # Submuestreo final
            nn.MaxPool2d(kernel_size=2, stride=2)         # -> 64 x 16 x 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------------------------------------------
# 3. Funcion de Entrenamiento y Evaluacion
# -------------------------------------------------------------
def train_and_evaluate(config, class_names, class_weights_tensor, device):
    exp_id = config["id"]
    model_name = config["model_name"]
    img_size = config["img_size"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    opt_type = config.get("optimizer", "adam")
    max_epochs = config.get("epochs", 20)
    patience = config.get("patience", 5)

    print("\n" + "=" * 60, flush=True)
    print(f"Iniciando {exp_id}: {config['description']}", flush=True)
    print(f"Resolucion: {img_size}x{img_size} | Batch: {batch_size} | Opt: {opt_type} (lr={lr})", flush=True)
    print("=" * 60, flush=True)

    # Transformaciones: Augmentation SOLO en train, sin volteo vertical
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(), # escala a [0.0, 1.0]
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = MushroomDataset(os.path.join(DATA_DIR, "train_10clases.csv"), LOCAL_DATASET_DIR, transform=train_transform)
    val_dataset = MushroomDataset(os.path.join(DATA_DIR, "val_10clases.csv"), LOCAL_DATASET_DIR, transform=eval_transform)
    test_dataset = MushroomDataset(os.path.join(DATA_DIR, "test_10clases.csv"), LOCAL_DATASET_DIR, transform=eval_transform)
    train_eval_dataset = MushroomDataset(os.path.join(DATA_DIR, "train_10clases.csv"), LOCAL_DATASET_DIR, transform=eval_transform)

    # num_workers=0 para total estabilidad en Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Instanciar modelo
    if model_name == "MLP":
        model = MushroomMLP(input_size=img_size*img_size*3, num_classes=len(class_names)).to(device)
    elif model_name == "CNN1":
        model = MushroomCNN1(num_classes=len(class_names)).to(device)
    elif model_name == "CNN2":
        model = MushroomCNN2(num_classes=len(class_names)).to(device)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    total_params = count_parameters(model)
    print(f"Total de parametros entrenables: {total_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    if opt_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_model_weights = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validacion
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoca [{epoch:02d}/{max_epochs:02d}] | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%", flush=True)

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"--> Early stopping activado en la epoca {epoch} (sin mejora en val_loss durante {patience} epocas).", flush=True)
                break

    training_time = time.time() - start_time
    print(f"Tiempo de entrenamiento: {training_time:.2f} s", flush=True)

    # Restaurar mejores pesos
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # 4. Grafico de progreso de Error General (Loss) y Acc
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    plt.title(f"{exp_id}: Progreso del Error General (Loss)")
    plt.xlabel("Época")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Acc", color="#2ca02c", lw=2)
    plt.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Val Acc", color="#d62728", lw=2, linestyle="--")
    plt.title(f"{exp_id}: Exactitud (Accuracy)")
    plt.xlabel("Época")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_chart_path = os.path.join(RESULTS_DIR, f"{exp_id}_progreso_loss.png")
    plt.savefig(loss_chart_path, dpi=200)
    plt.close()

    # 5. Evaluacion en Train y en Test (Métricas completas y matrices de confusión)
    def evaluate_dataset(loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        return np.array(all_labels), np.array(all_preds)

    y_train_true, y_train_pred = evaluate_dataset(train_eval_loader)
    y_test_true, y_test_pred = evaluate_dataset(test_loader)

    # Matriz de confusion y reportes
    cm_train = confusion_matrix(y_train_true, y_train_pred)
    cm_test = confusion_matrix(y_test_true, y_test_pred)

    rep_train = classification_report(y_train_true, y_train_pred, target_names=class_names, output_dict=True, zero_division=0)
    rep_test = classification_report(y_test_true, y_test_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Guardar Matrices de Confusion como graficos
    def plot_cm(cm, title, out_path):
        plt.figure(figsize=(8.5, 7))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title, fontsize=12, pad=12)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=8.5)
        plt.yticks(tick_marks, class_names, fontsize=8.5)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=8)
        plt.ylabel("Clase Real")
        plt.xlabel("Clase Predicha")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    cm_train_path = os.path.join(RESULTS_DIR, f"{exp_id}_matriz_train.png")
    cm_test_path = os.path.join(RESULTS_DIR, f"{exp_id}_matriz_test.png")
    plot_cm(cm_train, f"{exp_id}: Matriz de Confusión - Entrenamiento", cm_train_path)
    plot_cm(cm_test, f"{exp_id}: Matriz de Confusión - Prueba (Test)", cm_test_path)

    # Guardar reporte de metricas por clase en CSV
    metrics_rows = []
    for cls in class_names:
        metrics_rows.append({
            "entrenamiento": exp_id,
            "clase": cls,
            "precision_train": f"{rep_train[cls]['precision']*100:.2f}%",
            "recall_train": f"{rep_train[cls]['recall']*100:.2f}%",
            "f1_train": f"{rep_train[cls]['f1-score']*100:.2f}%",
            "precision_test": f"{rep_test[cls]['precision']*100:.2f}%",
            "recall_test": f"{rep_test[cls]['recall']*100:.2f}%",
            "f1_test": f"{rep_test[cls]['f1-score']*100:.2f}%",
            "soporte_test": int(rep_test[cls]["support"])
        })
    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(RESULTS_DIR, f"{exp_id}_metricas_clases.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)

    summary_entry = {
        "id": exp_id,
        "modelo": model_name,
        "descripcion": config["description"],
        "parametros": total_params,
        "resolucion": f"{img_size}x{img_size}",
        "batch_size": batch_size,
        "optimizador": opt_type,
        "lr": lr,
        "epocas_ejecutadas": len(history["epoch"]),
        "tiempo_seg": round(training_time, 2),
        "train_loss_final": round(history["train_loss"][-1], 4),
        "val_loss_final": round(history["val_loss"][-1], 4),
        "train_acc_pct": round(rep_train["accuracy"] * 100, 2),
        "val_acc_pct": round(history["val_acc"][-1] * 100, 2),
        "test_acc_pct": round(rep_test["accuracy"] * 100, 2),
        "test_f1_macro_pct": round(rep_test["macro avg"]["f1-score"] * 100, 2)
    }

    print(f"Resultados Finales {exp_id}:", flush=True)
    print(f"  Test Accuracy: {summary_entry['test_acc_pct']}% | Test F1 Macro: {summary_entry['test_f1_macro_pct']}%", flush=True)
    print(f"  Archivos generados: {loss_chart_path}, {cm_train_path}, {cm_test_path}, {metrics_csv_path}", flush=True)

    return summary_entry

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo de cómputo:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    df_train10 = pd.read_csv(os.path.join(DATA_DIR, "train_10clases.csv"))
    class_map = df_train10.drop_duplicates(subset=["clase_id"]).sort_values("clase_id")
    class_names = list(class_map["clase"])
    num_classes = len(class_names)

    counts = df_train10["clase_id"].value_counts().sort_index().values
    total_samples = len(df_train10)
    weights = total_samples / (num_classes * counts)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print("Class weights calculados:", [round(w, 2) for w in weights], flush=True)

    configs = [
        # Entrenamiento 1: MLP Base con Backpropagation
        {
            "id": "Entrenamiento_1_MLP",
            "model_name": "MLP",
            "description": "Perceptrón Multicapa (MLP) Base - 2 capas ocultas (256, 128) con ReLU",
            "img_size": 64,
            "batch_size": 64,
            "optimizer": "adam",
            "lr": 0.001,
            "epochs": 20,
            "patience": 5
        },
        # Entrenamiento 2: CNN Base de 1 Capa Convolucional
        {
            "id": "Entrenamiento_2_CNN_1Capa",
            "model_name": "CNN1",
            "description": "CNN 1 Capa Convolucional (32 filtros) + ReLU + MaxPool + Densa",
            "img_size": 128,
            "batch_size": 64,
            "optimizer": "adam",
            "lr": 0.001,
            "epochs": 20,
            "patience": 5
        },
        # Entrenamiento 3: CNN Profunda de 2 Capas Convolucionales
        {
            "id": "Entrenamiento_3_CNN_2Capas_Adam",
            "model_name": "CNN2",
            "description": "CNN 2 Capas Convolucionales (32->64 filtros crecientes) + Adam",
            "img_size": 128,
            "batch_size": 64,
            "optimizer": "adam",
            "lr": 0.001,
            "epochs": 20,
            "patience": 5
        },
        # Entrenamiento 4: CNN 2 Capas con SGD + Momentum
        {
            "id": "Entrenamiento_4_CNN_2Capas_SGD",
            "model_name": "CNN2",
            "description": "CNN 2 Capas Convolucionales - Comparativa con SGD (lr=0.005, momentum=0.9)",
            "img_size": 128,
            "batch_size": 64,
            "optimizer": "sgd",
            "lr": 0.005,
            "epochs": 20,
            "patience": 5
        },
        # Entrenamiento 5: CNN 2 Capas con menor batch y tasa fina
        {
            "id": "Entrenamiento_5_CNN_2Capas_Fino",
            "model_name": "CNN2",
            "description": "CNN 2 Capas Convolucionales - Ajuste fino (batch=32, lr=0.0005)",
            "img_size": 128,
            "batch_size": 32,
            "optimizer": "adam",
            "lr": 0.0005,
            "epochs": 20,
            "patience": 5
        }
    ]

    all_summaries = []
    for cfg in configs:
        summary = train_and_evaluate(cfg, class_names, class_weights_tensor, device)
        all_summaries.append(summary)

    df_summary = pd.DataFrame(all_summaries)
    summary_csv_path = os.path.join(OUTPUTS_DIR, "resumen_entrenamientos.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print("\n" + "=" * 60, flush=True)
    print("Resumen de todos los entrenamientos guardado en:", summary_csv_path, flush=True)
    print(df_summary[["id", "modelo", "parametros", "test_acc_pct", "test_f1_macro_pct", "tiempo_seg"]], flush=True)
    print("Todos los entrenamientos concluidos exitosamente.", flush=True)

if __name__ == "__main__":
    main()
