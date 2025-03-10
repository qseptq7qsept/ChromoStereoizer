import os
import sys
import torch
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                               QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QLineEdit, 
                               QMessageBox, QSlider, QGroupBox, QGridLayout)
from PySide6.QtGui import QPixmap, QImage
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

def apply_chromo_stereopsis(original_img: Image.Image, depth_img: Image.Image, threshold: float, feather: float) -> Image.Image:
    """
    Applies ChromoStereopsis processing:
      - Convert original image to grayscale.
      - For each pixel, compute a blending factor based on its depth value.
         Let t = threshold, f = feather (both in 0-255).
         If d <= t - f/2, blend = 0 (full blue);
         if d >= t + f/2, blend = 1 (full red);
         else, blend = (d - (t - f/2)) / f.
      - The resulting pixel color = (blend * gray, 0, (1 - blend) * gray).
    Returns a new RGB PIL Image.
    """
    gray = np.array(original_img.convert("L"), dtype=np.float32)
    depth_arr = np.array(depth_img, dtype=np.float32)
    h, w = gray.shape

    # Compute blending factor per pixel.
    half_feather = feather / 2.0
    blend = np.clip((depth_arr - (threshold - half_feather)) / (feather + 1e-6), 0, 1)

    # Create output image: for each pixel, red channel = blend * gray, blue channel = (1 - blend) * gray.
    output = np.zeros((h, w, 3), dtype=np.float32)
    output[..., 0] = blend * gray   # Red channel
    output[..., 2] = (1 - blend) * gray  # Blue channel

    output = np.clip(output, 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")

def pil_to_pixmap(im: Image.Image, max_size=(300, 300)) -> QPixmap:
    """Convert a PIL image to QPixmap for thumbnail display without modifying the original image."""
    if im.mode != "RGB":
        im = im.convert("RGB")
    # Create a copy for thumbnail creation so the original image remains unaltered
    thumbnail = im.copy()
    thumbnail.thumbnail(max_size, Image.Resampling.LANCZOS)
    data = thumbnail.tobytes("raw", "RGB")
    qimg = QImage(data, thumbnail.width, thumbnail.height, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromoStereoizer v1.0 -q7")
        self.resize(1200, 800)

        self.image_processor, self.model, self.device = load_model("vitl")
        self.output_folder = None
        self.original_img = None  # Loaded image
        self.depth_img = None     # Depth map

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout()
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        self.select_button = QPushButton("Select Input Image")
        self.select_button.clicked.connect(self.select_image)
        controls_layout.addWidget(self.select_button, 0, 0)

        self.folder_button = QPushButton("Select Output Folder")
        self.folder_button.clicked.connect(self.select_folder)
        controls_layout.addWidget(self.folder_button, 0, 1)

        self.folder_label = QLabel("No output folder selected")
        controls_layout.addWidget(self.folder_label, 0, 2)

        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("Enter output file name (without extension)")
        controls_layout.addWidget(self.filename_edit, 1, 0, 1, 2)

        # Slider for median threshold (% of 255)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        controls_layout.addWidget(QLabel("Median threshold (% of 255):"), 2, 0)
        controls_layout.addWidget(self.threshold_slider, 2, 1)

        # Feather slider (smooth blending region), as a percentage of 255.
        self.feather_slider = QSlider(Qt.Horizontal)
        self.feather_slider.setRange(0, 100)
        self.feather_slider.setValue(10)
        self.feather_slider.setTickInterval(5)
        self.feather_slider.setTickPosition(QSlider.TicksBelow)
        controls_layout.addWidget(QLabel("Feather width (% of 255):"), 3, 0)
        controls_layout.addWidget(self.feather_slider, 3, 1)

        # Two-step process buttons:
        self.process_depth_button = QPushButton("Process Depth")
        self.process_depth_button.clicked.connect(self.process_depth_only)
        controls_layout.addWidget(self.process_depth_button, 1, 2)

        self.process_chromo_button = QPushButton("Process ChromoStereopsis")
        self.process_chromo_button.clicked.connect(self.process_chromo)
        controls_layout.addWidget(self.process_chromo_button, 2, 2)

        # Preview areas for three outputs.
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

    def create_preview(self, title: str):
        layout = QVBoxLayout()
        label = QLabel(title)
        image_label = QLabel()
        # Do not force scaling; the thumbnail is already appropriately sized.
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
        # Show original image preview using a thumbnail copy.
        self.orig_preview["image_label"].setPixmap(pil_to_pixmap(self.original_img))
        self.orig_preview["image_label"].adjustSize()
        # Clear previous previews.
        self.depth_preview["image_label"].clear()
        self.chromo_preview["image_label"].clear()
        self.depth_img = None

    def process_depth_only(self):
        if self.original_img is None:
            QMessageBox.warning(self, "No Image", "Please select an input image first.")
            return
        try:
            # Process using the full-resolution image.
            self.depth_img = process_depth(self.original_img, self.image_processor, self.model, self.device)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Depth estimation failed:\n{e}")
            return
        # For preview, display a thumbnail version.
        self.depth_preview["image_label"].setPixmap(pil_to_pixmap(self.depth_img.convert("RGB")))
        self.depth_preview["image_label"].adjustSize()

    def process_chromo(self):
        if self.original_img is None:
            QMessageBox.warning(self, "No Image", "Please select an input image first.")
            return
        # Ensure depth has been processed; if not, process it.
        if self.depth_img is None:
            try:
                self.depth_img = process_depth(self.original_img, self.image_processor, self.model, self.device)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Depth estimation failed:\n{e}")
                return

        # Get threshold from slider (as percentage of 255).
        threshold = (self.threshold_slider.value() / 100.0) * 255
        # Get feather width from slider (as percentage of 255).
        feather = (self.feather_slider.value() / 100.0) * 255

        try:
            chromo_img = apply_chromo_stereopsis(self.original_img, self.depth_img, threshold, feather)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ChromoStereopsis processing failed:\n{e}")
            return

        self.chromo_preview["image_label"].setPixmap(pil_to_pixmap(chromo_img))
        self.chromo_preview["image_label"].adjustSize()

        # Save processed chromo image if output folder is selected.
        if self.output_folder:
            base_name = self.filename_edit.text().strip() or "processed_image"
            out_path = os.path.join(self.output_folder, base_name + ".png")
            try:
                chromo_img.save(out_path)
                QMessageBox.information(self, "Saved", f"Processed image saved to:\n{out_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save processed image:\n{e}")
        else:
            QMessageBox.warning(self, "Output Folder", "No output folder selected. Processed image not saved.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
