"""
train.py
========
Step 2 of 3 in the pipeline.

Trains a YOLOv8 classification model on your PlantVillage dataset.
Replaces the original Google Colab notebook entirely.

Requirements:
    - Run prepare_dataset.py first to create the dataset/ folder
    - pip install ultralytics torch torchvision

Usage:
    python train.py                          # uses all defaults
    python train.py --data dataset           # custom dataset path
    python train.py --model yolov8s-cls.pt   # use a larger model
    python train.py --epochs 100 --batch 32  # custom hyperparams
    python train.py --evaluate               # evaluate after training
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def check_dataset(data_dir: str):
    """Verify that the dataset folder has the expected structure."""
    data = Path(data_dir)
    if not data.exists():
        print(f"\n[ERROR] Dataset folder '{data_dir}' not found.")
        print("  Please run: python prepare_dataset.py --src PlantVillage --dst dataset\n")
        sys.exit(1)

    for split in ['train', 'val']:
        split_path = data / split
        if not split_path.exists():
            print(f"\n[ERROR] Missing '{split}' folder inside {data_dir}.")
            print("  Please run: python prepare_dataset.py --src PlantVillage --dst dataset\n")
            sys.exit(1)

    classes = [d.name for d in (data / 'train').iterdir() if d.is_dir()]
    if not classes:
        print(f"\n[ERROR] No class folders found in {data_dir}/train/")
        sys.exit(1)

    print(f"\n  Dataset verified: {len(classes)} classes found in '{data_dir}'")
    return len(classes)


def train(data_dir='dataset', model_name='yolov8n-cls.pt', epochs=50,
          batch=32, imgsz=224, device='', workers=4,
          project='runs/classify', name='crop_disease',
          evaluate=False, resume=False):

    # Lazy import so error messages are clean if not installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics not installed. Run: pip install ultralytics\n")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Crop Disease YOLOv8 Classification Training")
    print(f"{'='*60}")

    # Verify dataset
    num_classes = check_dataset(data_dir)

    print(f"  Model      : {model_name}")
    print(f"  Dataset    : {data_dir}  ({num_classes} classes)")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch      : {batch}")
    print(f"  Image size : {imgsz}x{imgsz}")
    print(f"  Device     : {'auto (GPU if available)' if device == '' else device}")
    print(f"{'='*60}\n")

    # Load model
    if resume:
        # Resume from last checkpoint
        last_weights = Path(project) / name / 'weights' / 'last.pt'
        if not last_weights.exists():
            print(f"[ERROR] Cannot resume: no checkpoint found at {last_weights}")
            sys.exit(1)
        model = YOLO(str(last_weights))
        print(f"  Resuming from: {last_weights}\n")
    else:
        model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_dir,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device if device else None,
        workers=workers,
        project=project,
        name=name,
        verbose=True,
        plots=True,
        exist_ok=resume      # allow overwriting if resuming
    )

    # Print training summary
    best_weights = Path(project) / name / 'weights' / 'best.pt'
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best model saved to: {best_weights.resolve()}")
    print(f"{'='*60}\n")

    # Copy best.pt to project root for easy access by app.py
    if best_weights.exists():
        shutil.copy2(best_weights, 'best.pt')
        print(f"  Copied best.pt to project root (for app.py to use)\n")

    # Plot training curves
    results_csv = Path(project) / name / 'results.csv'
    if results_csv.exists():
        plot_training_curves(results_csv, save_path=Path(project) / name / 'training_analysis.png')

    # Optional evaluation
    if evaluate:
        run_evaluation(model, data_dir, project, name)

    return model


def plot_training_curves(results_csv: Path, save_path: Path = None):
    """Plot train/val accuracy and loss curves from results.csv."""
    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('YOLOv8 Classification Training Results', fontsize=14, fontweight='bold')

        # Try to find accuracy and loss columns (column names differ by ultralytics version)
        acc_train_col = next((c for c in df.columns if 'train' in c.lower() and 'acc' in c.lower()), None)
        acc_val_col   = next((c for c in df.columns if 'val' in c.lower() and 'acc' in c.lower()), None)
        loss_train_col = next((c for c in df.columns if 'train' in c.lower() and 'loss' in c.lower()), None)
        loss_val_col   = next((c for c in df.columns if 'val' in c.lower() and 'loss' in c.lower()), None)

        epochs = range(1, len(df) + 1)

        # Accuracy plot
        ax1 = axes[0]
        if acc_train_col:
            ax1.plot(epochs, df[acc_train_col], label='Train Accuracy', color='blue', linewidth=2)
        if acc_val_col:
            ax1.plot(epochs, df[acc_val_col], label='Val Accuracy', color='orange', linewidth=2)
        ax1.set_title('Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss plot
        ax2 = axes[1]
        if loss_train_col:
            ax2.plot(epochs, df[loss_train_col], label='Train Loss', color='blue', linewidth=2)
        if loss_val_col:
            ax2.plot(epochs, df[loss_val_col], label='Val Loss', color='orange', linewidth=2)
        ax2.set_title('Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Training curves saved to: {save_path}")

        plt.show()

    except Exception as e:
        print(f"  [WARN] Could not plot training curves: {e}")


def run_evaluation(model, data_dir: str, project: str, name: str):
    """Run evaluation on the test split and print classification report."""
    test_dir = Path(data_dir) / 'test'
    if not test_dir.exists():
        print("  [INFO] No test/ split found, skipping evaluation.")
        return

    print(f"\n{'='*60}")
    print(f"  Running Evaluation on Test Set")
    print(f"{'='*60}\n")

    try:
        from ultralytics import YOLO
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix
        from PIL import Image

        model_path = Path(project) / name / 'weights' / 'best.pt'
        eval_model = YOLO(str(model_path))

        class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

        y_true = []
        y_pred = []

        print("  Running inference on test images...")
        for class_idx, class_name in enumerate(class_names):
            class_dir = test_dir / class_name
            img_files = list(class_dir.glob('*'))
            img_files = [f for f in img_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

            for img_path in img_files:
                results = eval_model(str(img_path), verbose=False)
                pred_class = int(results[0].probs.top1)
                pred_name  = eval_model.names[pred_class]
                y_true.append(class_name)
                y_pred.append(pred_name)

        # Only show classes that appear in test set
        present_classes = sorted(set(y_true))

        print("\n  Classification Report:")
        print("  " + "-" * 56)
        report = classification_report(y_true, y_pred, labels=present_classes, zero_division=0)
        for line in report.splitlines():
            print("  " + line)

        # Save report
        report_path = Path(project) / name / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("Crop Disease Classification - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print(f"\n  Report saved to: {report_path}")

    except ImportError:
        print("  [WARN] sklearn not installed. Run: pip install scikit-learn")
    except Exception as e:
        print(f"  [ERROR] Evaluation failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 classification model on PlantVillage dataset')
    parser.add_argument('--data',     type=str,   default='dataset',          help='Path to dataset folder (default: dataset)')
    parser.add_argument('--model',    type=str,   default='yolov8n-cls.pt',   help='YOLOv8 model: yolov8n-cls.pt | yolov8s-cls.pt | yolov8m-cls.pt (default: yolov8n-cls.pt)')
    parser.add_argument('--epochs',   type=int,   default=50,                 help='Number of training epochs (default: 50)')
    parser.add_argument('--batch',    type=int,   default=32,                 help='Batch size (default: 32)')
    parser.add_argument('--imgsz',    type=int,   default=224,                help='Image size (default: 224)')
    parser.add_argument('--device',   type=str,   default='',                 help='Device: 0 for GPU, cpu for CPU, empty for auto (default: auto)')
    parser.add_argument('--workers',  type=int,   default=4,                  help='Dataloader workers (default: 4)')
    parser.add_argument('--project',  type=str,   default='runs/classify',    help='Output folder (default: runs/classify)')
    parser.add_argument('--name',     type=str,   default='crop_disease',     help='Run name (default: crop_disease)')
    parser.add_argument('--evaluate', action='store_true',                     help='Run evaluation on test set after training')
    parser.add_argument('--resume',   action='store_true',                     help='Resume training from last checkpoint')
    args = parser.parse_args()

    train(
        data_dir=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        evaluate=args.evaluate,
        resume=args.resume
    )
