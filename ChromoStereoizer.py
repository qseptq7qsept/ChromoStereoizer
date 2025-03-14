import os
import sys
import torch
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                               QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QLineEdit, 
                               QMessageBox, QSlider, QGroupBox)
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor
from PySide6.QtCore import Qt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def load_model(encoder: str = "vitl"):
    # For simplicity, we use the small variant.
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return image_processor, model, device

def process_depth(image: Image.Image, image_processor, model, device):
    """Run depth estimation and return a depth map (PIL Image, mode "L")."""
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )
    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min() + 1e-6)
    depth = depth.detach().cpu().numpy() * 255
    depth_img = Image.fromarray(depth.astype("uint8"))
    return depth_img

def apply_chromo_stereopsis(
    original_img: Image.Image,
    depth_img: Image.Image,
    threshold: float,
    feather: float,
    black_level: float,
    white_level: float
) -> Image.Image:
    """
    Chromostereopsis with a "Levels" approach for grayscale:
    
    1) Convert original to grayscale (0..255).
    2) Apply a levels adjustment:
         adjusted_gray = clamp( (gray - black_level)/(white_level - black_level)*255 , 0..255 )
    3) Convert adjusted_gray to [0..1] for color multiplication.
    4) Compute blend from depth:
         - blend=0 => fully blue
         - blend=1 => fully red
    5) Red channel = blend * adjusted_gray_01 * 255
       Blue channel = (1 - blend) * adjusted_gray_01 * 255
       Green channel = 0
    """
    # 1) Convert to grayscale
    gray = np.array(original_img.convert("L"), dtype=np.float32)

    # 2) Levels adjustment
    # Ensure we don't divide by zero if black_level == white_level
    denom = (white_level - black_level) if (white_level > black_level) else 1e-6
    adjusted_gray = (gray - black_level) / denom * 255.0
    adjusted_gray = np.clip(adjusted_gray, 0, 255)

    # Convert to [0..1] for color multiplication
    adjusted_gray_01 = adjusted_gray / 255.0

    # 3) Depth-based blend factor
    depth_arr = np.array(depth_img, dtype=np.float32)
    half_feather = feather / 2.0
    blend = np.clip((depth_arr - (threshold - half_feather)) / (feather + 1e-6), 0, 1)

    # 4) Red / Blue channels
    red = blend * adjusted_gray_01 * 255.0
    blue = (1.0 - blend) * adjusted_gray_01 * 255.0

    # Create the output image
    h, w = gray.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[..., 0] = np.clip(red, 0, 255).astype(np.uint8)
    output[..., 2] = np.clip(blue, 0, 255).astype(np.uint8)

    return Image.fromarray(output, mode="RGB")

def pil_to_pixmap(im: Image.Image, max_size=(300, 300)) -> QPixmap:
    if im.mode != "RGB":
        im = im.convert("RGB")
    thumbnail = im.copy()
    thumbnail.thumbnail(max_size, Image.Resampling.LANCZOS)
    data = thumbnail.tobytes("raw", "RGB")
    bytes_per_line = thumbnail.width * 3
    qimg = QImage(data, thumbnail.width, thumbnail.height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromoStereoizer - v1.2 -q7")
        self.resize(1200, 800)

        self.image_processor, self.model, self.device = load_model("vitl")
        self.output_folder = None
        self.original_img = None
        self.depth_img = None
        self.chromo_img = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # Row 1: select image, select folder
        top_row = QHBoxLayout()
        self.select_button = QPushButton("Select Input Image")
        self.select_button.clicked.connect(self.select_image)
        top_row.addWidget(self.select_button)

        self.folder_button = QPushButton("Select Output Folder")
        self.folder_button.clicked.connect(self.select_folder)
        top_row.addWidget(self.folder_button)

        self.folder_label = QLabel("No output folder selected")
        top_row.addWidget(self.folder_label)
        controls_layout.addLayout(top_row)

        # Row 2: filename
        row_two = QHBoxLayout()
        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("Enter output file name (without extension)")
        row_two.addWidget(self.filename_edit)
        controls_layout.addLayout(row_two)

        # Threshold slider
        threshold_row = QHBoxLayout()
        threshold_desc = QLabel("Median threshold (% of 255):")
        threshold_row.addWidget(threshold_desc)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.threshold_slider.valueChanged.connect(lambda _: self.update_chromo_preview())
        threshold_row.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel("50")
        threshold_row.addWidget(self.threshold_value_label)
        controls_layout.addLayout(threshold_row)

        # Feather slider
        feather_row = QHBoxLayout()
        feather_desc = QLabel("Feather width (% of 255):")
        feather_row.addWidget(feather_desc)
        self.feather_slider = QSlider(Qt.Horizontal)
        self.feather_slider.setRange(0, 100)
        self.feather_slider.setValue(10)
        self.feather_slider.setTickInterval(5)
        self.feather_slider.setTickPosition(QSlider.TicksBelow)
        self.feather_slider.valueChanged.connect(self.update_feather_label)
        self.feather_slider.valueChanged.connect(lambda _: self.update_chromo_preview())
        feather_row.addWidget(self.feather_slider)
        self.feather_value_label = QLabel("10")
        feather_row.addWidget(self.feather_value_label)
        controls_layout.addLayout(feather_row)

        # Black Level slider (0..255)
        black_row = QHBoxLayout()
        black_desc = QLabel("Black Level (Default = 0):")
        black_row.addWidget(black_desc)
        self.black_slider = QSlider(Qt.Horizontal)
        self.black_slider.setRange(0, 255)
        self.black_slider.setValue(0)
        self.black_slider.setTickInterval(16)
        self.black_slider.setTickPosition(QSlider.TicksBelow)
        self.black_slider.valueChanged.connect(self.update_black_label)
        self.black_slider.valueChanged.connect(lambda _: self.update_chromo_preview())
        black_row.addWidget(self.black_slider)
        self.black_value_label = QLabel("0")
        black_row.addWidget(self.black_value_label)
        controls_layout.addLayout(black_row)

        # White Level slider (0..255)
        white_row = QHBoxLayout()
        white_desc = QLabel("White Level (Default = 255):")
        white_row.addWidget(white_desc)
        self.white_slider = QSlider(Qt.Horizontal)
        self.white_slider.setRange(0, 255)
        self.white_slider.setValue(255)
        self.white_slider.setTickInterval(16)
        self.white_slider.setTickPosition(QSlider.TicksBelow)
        self.white_slider.valueChanged.connect(self.update_white_label)
        self.white_slider.valueChanged.connect(lambda _: self.update_chromo_preview())
        white_row.addWidget(self.white_slider)
        self.white_value_label = QLabel("255")
        white_row.addWidget(self.white_value_label)
        controls_layout.addLayout(white_row)

        # Save Chromostereopsis
        process_chromo_row = QHBoxLayout()
        self.process_chromo_button = QPushButton("Save Chromostereopsis")
        self.process_chromo_button.clicked.connect(self.save_chromo)
        process_chromo_row.addWidget(self.process_chromo_button)
        controls_layout.addLayout(process_chromo_row)

        # Preview group
        preview_group = QGroupBox("Previews")
        preview_layout = QHBoxLayout()
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        self.orig_preview = self.create_preview("Original Image")
        self.depth_preview = self.create_preview("Depth Map")
        self.chromo_preview = self.create_preview("ChromoStereopsis")
        preview_layout.addLayout(self.orig_preview["layout"])
        preview_layout.addLayout(self.depth_preview["layout"])
        preview_layout.addLayout(self.chromo_preview["layout"])

    def update_threshold_label(self, value):
        self.threshold_value_label.setText(str(value))

    def update_feather_label(self, value):
        self.feather_value_label.setText(str(value))

    def update_black_label(self, value):
        self.black_value_label.setText(str(value))

    def update_white_label(self, value):
        self.white_value_label.setText(str(value))

    def create_preview(self, title: str):
        layout = QVBoxLayout()
        label = QLabel(title)
        image_label = QLabel()
        image_label.setScaledContents(False)
        layout.addWidget(label)
        layout.addWidget(image_label)
        return {"layout": layout, "image_label": image_label}

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if folder:
            self.output_folder = folder
            self.folder_label.setText(f"Output Folder: {folder}")
        else:
            self.output_folder = None
            self.folder_label.setText("No output folder selected")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        try:
            self.original_img = Image.open(file_path).convert("RGB")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open image:\n{e}")
            self.original_img = None
            return

        self.orig_preview["image_label"].setPixmap(pil_to_pixmap(self.original_img))
        self.orig_preview["image_label"].adjustSize()

        self.depth_preview["image_label"].clear()
        self.chromo_preview["image_label"].clear()
        self.depth_img = None
        self.chromo_img = None

        self.update_chromo_preview()

    def update_chromo_preview(self):
        if self.original_img is None:
            self.chromo_preview["image_label"].clear()
            self.chromo_img = None
            return

        if self.depth_img is None:
            try:
                self.depth_img = process_depth(self.original_img, self.image_processor, self.model, self.device)
                self.depth_preview["image_label"].setPixmap(pil_to_pixmap(self.depth_img.convert("RGB")))
                self.depth_preview["image_label"].adjustSize()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Depth estimation failed:\n{e}")
                return

        threshold = (self.threshold_slider.value() / 100.0) * 255
        feather = (self.feather_slider.value() / 100.0) * 255
        black_level = float(self.black_slider.value())
        white_level = float(self.white_slider.value())

        try:
            self.chromo_img = apply_chromo_stereopsis(
                self.original_img,
                self.depth_img,
                threshold,
                feather,
                black_level,
                white_level
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ChromoStereopsis processing failed:\n{e}")
            self.chromo_img = None
            return

        self.chromo_preview["image_label"].setPixmap(pil_to_pixmap(self.chromo_img))
        self.chromo_preview["image_label"].adjustSize()

    def save_chromo(self):
        if self.chromo_img is None:
            QMessageBox.warning(self, "No Processed Image", "ChromoStereopsis preview not available.")
            return
        if self.output_folder:
            base_name = self.filename_edit.text().strip() or "processed_image"
            out_path = os.path.join(self.output_folder, base_name + ".png")
            try:
                self.chromo_img.save(out_path)
                QMessageBox.information(self, "Saved", f"Processed image saved to:\n{out_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save processed image:\n{e}")
        else:
            QMessageBox.warning(self, "Output Folder", "No output folder selected. Processed image not saved.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Simple dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(208, 42, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(208, 42, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
