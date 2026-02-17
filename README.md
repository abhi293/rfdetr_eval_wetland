# RF-DETR Bird Detection on Visual Wetland Dataset

This project evaluates different versions of the RF-DETR (Receptive Field Aware DETR) object detection models on a custom bird detection dataset. The Visual Wetland dataset was obtained from Zenodo ([https://zenodo.org/records/15696105](https://zenodo.org/records/15696105)) and contains annotated videos of various bird species in wetland environments.

## 🎯 Project Overview

Since RF-DETR models were originally trained on 80 COCO classes for general object detection, we leverage the pre-trained "bird" class (ID 16) to identify birds in the Visual Wetland dataset. This project includes:

- **Dataset Preparation**: Converting Visual Wetland video annotations to COCO format
- **Model Evaluation**: Testing RF-DETR Nano, Medium, and Base models on bird detection
- **Performance Analysis**: Computing mAP, Precision, Recall, and F1 scores
- **Video Reconstruction**: Annotating videos with predicted and ground truth bounding boxes

## 📊 Dataset Information

**Source**: [Visual Wetland Dataset - Zenodo](https://zenodo.org/records/15696105)

The original dataset contains:
- Videos of various bird species in wetland environments
- Frame-level annotations with bounding boxes
- Species identifications and behavior labels
- Split information (train/validation/test)

**Dataset Processing**:
- Frames extracted from videos with downsampling (default: every 10th frame)
- Annotations converted to COCO JSON format (`_annotations.coco.json`)
- Images organized into `train/`, `valid/`, and `test/` folders
- Bird class mapped to RF-DETR's COCO class ID 16

## 📁 Project Structure

```
RF_DETR_Wetland/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── rf-detr-nano.pth                   # Pre-trained RF-DETR Nano model weights
├── rf-detr-medium.pth                 # Pre-trained RF-DETR Medium model weights
├── rf-detr-base.pth                   # Pre-trained RF-DETR Base model weights
│
├── rf_detr/                           # Original Visual Wetland dataset
│   ├── videos/                        # Original video files (.mp4)
│   ├── _annotations.coco.json         # Generated COCO annotations
│   ├── bounding_boxes.csv             # Raw bounding box data
│   ├── species_ID.csv                 # Species ID mappings
│   ├── behaviors_ID.csv               # Behavior ID mappings
│   ├── crops.csv                      # Crop information
│   ├── splits.json                    # Train/val/test splits
│   ├── prepare_dataset.py             # Dataset preparation script
│   └── clone_rf_detr_dataset.py       # Dataset subset creation utility
│
├── dataset/                           # Processed dataset for RF-DETR
│   ├── train/
│   │   ├── images/                    # Training images
│   │   └── _annotations.coco.json     # Training annotations
│   ├── valid/
│   │   ├── images/                    # Validation images
│   │   └── _annotations.coco.json     # Validation annotations
│   └── test/
│       ├── images/                    # Test images
│       └── _annotations.coco.json     # Test annotations
│
├── annotate_info.py                   # Generate COCO annotations from CSV
├── json_to_excel.py                   # Convert JSON results to Excel
│
├── main.ipynb                         # Training notebook (RF-DETR Base)
├── nano.ipynb                         # Inference script (RF-DETR Nano)
├── evual.ipynb                        # Evaluation script (RF-DETR Base)
├── results.ipynb                      # Results analysis & visualization
├── reconstruct_video.ipynb            # Video annotation & evaluation script
│
├── results/                           # Analysis outputs
│   ├── class_metrics_test.csv         # Per-class metrics (test set)
│   ├── class_metrics_valid.csv        # Per-class metrics (validation set)
│   ├── metrics_by_epoch.csv           # Training metrics over time
│   └── results_full.json              # Complete results in JSON
│
├── evaluation_only/                   # Headless evaluation outputs
│   ├── evaluation_report_nano_full.csv
│   ├── evaluation_report_nano_20.csv
│   ├── evaluation_report_medium_full.csv
│   └── evaluation_report_base_10.csv
│
├── test_results_final/                # Image-based test results
│   ├── detection_summary.csv
│   └── detection_summary.json
│
├── annotated_videos/                  # Video outputs with bounding boxes
│   └── [video_name]_annotated.mp4
│
└── content/                           # Training and evaluation artifacts
    ├── rfdetr_results/                # Training checkpoints
    │   ├── checkpoint_best_ema.pth
    │   ├── checkpoint_best_regular.pth
    │   ├── checkpoint.pth
    │   ├── log.txt
    │   └── results.json
    └── eval_results/                  # Evaluation outputs
        ├── COCO_Baseline_Summary.csv
        └── Species_Baseline_Effect.csv
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training/inference)
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/abhi293/rfdetr_eval_wetland>
   cd RF_DETR_Wetland
   ```

2. **Create a Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Weights**
   
   Download the RF-DETR model weights and place them in the project root:
   - `rf-detr-nano.pth`
   - `rf-detr-medium.pth`
   - `rf-detr-base.pth`
   
   These are automatically downloaded by the `rfdetr` library on first use.
   
   Or force download using the below script (change the model version as requirement):
   ```bash
   from rfdetr.main import download_pretrain_weights
   try:
        print("Attempting to force-download Base weights...")
        download_pretrain_weights("rf-detr-base.pth", redownload=True)
        print("Download successful!")
    except Exception as e:
    print(f"Download failed: {e}")
   ```
5. **Download Visual Wetland Dataset**
   
   Download the dataset from [Zenodo](https://zenodo.org/records/15696105) and extract it to the `rf_detr/` folder with the following structure:
   ```
   rf_detr/
   ├── videos/
   ├── bounding_boxes.csv
   ├── species_ID.csv
   ├── behaviors_ID.csv
   ├── crops.csv
   └── splits.json
   ```

## 📝 Usage Guide

### 1. Dataset Preparation

**Generate COCO Annotations from Visual Wetland Dataset**

```bash
cd rf_detr
python prepare_dataset.py \
    --bounding_csv bounding_boxes.csv \
    --videos_dir videos \
    --splits_json splits.json \
    --species_csv species_ID.csv \
    --output_dir ../dataset \
    --downsample 10 \
    --zip
```

**Parameters:**
- `--bounding_csv`: Path to bounding boxes CSV file
- `--videos_dir`: Directory containing videos
- `--splits_json`: JSON file with train/val/test splits
- `--species_csv`: CSV file with species IDs
- `--output_dir`: Output directory for processed dataset
- `--downsample`: Frame sampling rate (e.g., 10 = every 10th frame)
- `--zip`: Create zip archives for each split
- `--image_ext`: Output image format (default: jpg)
- `--overwrite`: Overwrite existing images

**Alternative: Generate Annotations from CSV** (if you already have images)

```bash
python annotate_info.py
```

This script reads `bounding_boxes.csv` and `species_ID.csv` from the `rf_detr/` folder and generates `_annotations.coco.json`.

### 2. Model Inference

**RF-DETR Nano - Image-based Inference**

Open `nano.ipynb` and run the cells. This script:
- Loads the RF-DETR Nano model with pre-trained weights
- Processes images from the test set
- Filters detections for bird class (ID 16)
- Saves annotated images to `test_results_final/`

**Configuration:**
```python
model = RFDETRNano(
    patch_size=16,
    positional_encoding_size=24,
    resolution=384,
    out_feature_indexes=[3, 6, 9, 12],
    num_windows=2,
    dec_layers=2,
    pretrain_weights="rf-detr-nano.pth"
)
```

### 3. Video Reconstruction with Bounding Boxes

**Annotate Videos with Predictions and Ground Truth**

Open `reconstruct_video.ipynb` and configure:

```python
SOURCE_FOLDER = r"D:\Projects\RF_DETR_Wetland\rf_detr\videos"
TARGET_FOLDER = r"D:\Projects\RF_DETR_Wetland\annotated_videos"
ANNOTATIONS_PATH = r"D:\Projects\RF_DETR_Wetland\rf_detr\_annotations.coco.json"

START_INDEX = 0   # Starting video index
END_INDEX = 50    # Ending video index
```

**Features:**
- Processes videos in specified range
- Overlays predicted bounding boxes on frames
- Compares predictions with ground truth
- Computes mAP, Precision, Recall, and F1 score
- Saves annotated videos to `annotated_videos/`
- Generates performance reports (JSON & CSV)

**Output:**
- `[video_name]_annotated.mp4` - Video with bounding boxes
- `range_eval_results.json` - Performance metrics
- `range_eval_summary.csv` - Summary statistics

### 4. Headless Performance Evaluation

**Evaluate Models WITHOUT Video Reconstruction** (faster)

Open `reconstruct_video.ipynb` and run the evaluation cells (Cell #3, #4, #5):

**RF-DETR Nano Evaluation:**
```python
model = RFDETRNano(
    patch_size=16, 
    positional_encoding_size=24, 
    resolution=384,
    pretrain_weights="rf-detr-nano.pth"
)
START_IDX = 0
STOP_IDX = 178  # All videos
```

**RF-DETR Medium Evaluation:**
```python
model = RFDETRMedium(
    patch_size=16, 
    resolution=416,
    pretrain_weights="rf-detr-medium.pth"
)
```

**RF-DETR Base Evaluation:**
```python
model = RFDETRBase(
    pretrain_weights="rf-detr-base.pth"
)
```

**Output:**
- `evaluation_report_[model]_[range].csv` - Detailed metrics per IoU threshold

### 5. Results Analysis

**Analyze Training Logs and Generate Plots**

Open `results.ipynb` and run all cells. This script:
- Parses training logs from `rfdetr_results/log.txt`
- Extracts per-class metrics from `results.json`
- Generates CSV files for analysis
- Creates plots (loss curves, mAP over epochs, per-class performance)

**Outputs:**
- `results/metrics_by_epoch.csv` - Epoch-wise metrics
- `results/class_metrics_valid.csv` - Validation set class metrics
- `results/class_metrics_test.csv` - Test set class metrics
- `results/loss_per_epoch.png` - Loss visualization
- `results/map_over_epochs.png` - mAP progression
- `results/per_class_map_valid.png` - Per-class mAP bar chart

**Convert Results to Excel**

```bash
python json_to_excel.py
```

Converts `results/results_full.json` to `results/results_summary.xlsx` with separate sheets for validation and test sets.

### 6. Training (Optional)

**Fine-tune RF-DETR Models on Visual Wetland Dataset**

Open `main.ipynb` and configure training:

```python
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir="dataset",
    output_dir="rfdetr_results",
    epochs=15,
    batch_size=4,
    grad_accum_steps=4,
    amp=True,
    num_workers=2,
    checkpoint_interval=1,
    resume="rfdetr_results/checkpoint_best_ema.pth"  # Resume from checkpoint
)
```

**Training Parameters:**
- `dataset_dir`: Path to dataset folder with train/valid/test splits
- `output_dir`: Directory to save checkpoints and logs
- `epochs`: Number of training epochs
- `batch_size`: Images per batch
- `grad_accum_steps`: Gradient accumulation steps
- `amp`: Enable automatic mixed precision
- `num_workers`: Data loading workers
- `checkpoint_interval`: Save checkpoint every N epochs
- `resume`: Path to checkpoint for resuming training

## 📊 Performance Metrics

The evaluation scripts compute the following metrics:

| Metric | Description |
|--------|-------------|
| **mAP@50:95** | Mean Average Precision at IoU thresholds 0.50 to 0.95 |
| **mAP@50** | Mean Average Precision at IoU threshold 0.50 |
| **mAP@75** | Mean Average Precision at IoU threshold 0.75 |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | Harmonic mean of Precision and Recall |

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in training/inference scripts
   - Lower image resolution
   - Enable gradient accumulation

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Model Weights Not Found**
   - Ensure `.pth` files are in the project root
   - Check file names match exactly (case-sensitive)

4. **COCO Annotations Mismatch**
   - Verify `_annotations.coco.json` exists in each split folder
   - Ensure image filenames match annotation entries
   - Re-run `prepare_dataset.py` if issues persist

## 📚 Key Scripts Reference

| Script | Purpose |
|--------|---------|
| `annotate_info.py` | Generate COCO annotations from CSV files |
| `prepare_dataset.py` | Extract frames from videos and create dataset |
| `json_to_excel.py` | Convert JSON results to Excel format |
| `nano.ipynb` | Run inference with RF-DETR Nano model |
| `evual.ipynb` | Evaluate RF-DETR Base model |
| `reconstruct_video.ipynb` | Annotate videos and compute performance metrics |
| `results.ipynb` | Analyze and visualize results |
| `main.ipynb` | Training script for RF-DETR models |

## 📖 Citation

If you use this project or the Visual Wetland dataset, please cite:

```bibtex
@dataset{visual_wetland_2024,
  title={Visual Wetland Dataset},
  author={[Authors]},
  year={2024},
  publisher={Zenodo},
  doi={10.5281/zenodo.15696105},
  url={https://zenodo.org/records/15696105}
}
```

## 📄 License

This project is for research and educational purposes. Please refer to the RF-DETR library license and Visual Wetland dataset license for usage restrictions.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: February 2026
