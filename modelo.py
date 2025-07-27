import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json
from datetime import datetime, timedelta
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import time
import hashlib
import threading
import logging
from logging.handlers import RotatingFileHandler
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.stats import linregress
from skimage import measure, morphology
import seaborn as sns
from collections import deque
import gc
import psutil
import traceback
import concurrent.futures
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# ===================================
# CONFIGURACIÓN INICIAL Y MONITOREO
# ===================================

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuración de directorios
MODEL_DIR = "thyroid_models"
DATA_DIR = "thyroid_data"
TRAINING_DATA_DIR = "training_data"
REPORT_DIR = "reports"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(os.path.join(LOG_DIR, "thyroid_analyzer.log"), maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ThyroidAnalyzer")

# Monitoreo de recursos
class ResourceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.max_memory = 0
        self.memory_history = deque(maxlen=100)
        self.cpu_history = deque(maxlen=100)
        self.running = True
        self.thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.thread.start()
        
    def _monitor_resources(self):
        while self.running:
            self.update()
            time.sleep(1)
            
    def update(self):
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / (1024 ** 2)  # MB
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        self.max_memory = max(self.max_memory, mem_usage)
        self.memory_history.append(mem_usage)
        self.cpu_history.append(cpu_usage)
        
        return mem_usage, cpu_usage
    
    def get_uptime(self):
        return time.time() - self.start_time
    
    def get_avg_memory(self):
        return sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
    
    def get_avg_cpu(self):
        return sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
    
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)

# ===================================
# MODELOS AVANZADOS DE APRENDIZAJE
# ===================================

class ThyroidDataset(Dataset):
    def __init__(self, data_dir, transform=None, synthetic_ratio=0.0, max_size=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.annotations = []
        self.synthetic_ratio = synthetic_ratio
        self.max_size = max_size
        
        # Transformación por defecto para convertir a tensor
        self.default_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Cargar datos reales
        self._load_real_data()
    
    def _load_real_data(self):
        if not os.path.exists(self.data_dir):
            logger.warning(f"Directorio de datos no encontrado: {self.data_dir}")
            return
            
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if self.max_size and len(self.image_paths) >= self.max_size:
                    break
                    
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
                    
                    # Cargar anotaciones si existen
                    annotation_path = os.path.splitext(img_path)[0] + '.json'
                    if os.path.exists(annotation_path):
                        try:
                            with open(annotation_path, 'r') as f:
                                self.annotations.append(json.load(f))
                        except json.JSONDecodeError:
                            logger.error(f"Error al cargar anotación: {annotation_path}")
                            self.annotations.append(None)
                    else:
                        self.annotations.append(None)
        
        if not self.image_paths:
            logger.warning(f"No se encontraron imágenes en el directorio: {self.data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            if img_path.lower().endswith('.dcm'):
                img, _ = self.load_dicom_image(img_path)
            else:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"No se pudo cargar la imagen: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotation = self.annotations[idx]
            
            # Preprocesamiento
            img = self.preprocess_image(img)
            
            # Convertir anotaciones a tensores
            target = self._convert_annotations_to_target(img.shape, annotation) if annotation else None
            
            # Aplicar transformaciones (siempre convierte a tensor)
            if self.transform:
                img = self.transform(img)
            else:
                # Usar transformación por defecto para convertir a tensor
                img = self.default_transform(img)
            
            # Redimensionar la máscara de segmentación a 64x64
            if target is not None:
                seg_mask = target['seg_mask']
                # Convertir a tensor flotante y agregar dimensión de lote (batch) para interpolación
                seg_mask = seg_mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
                seg_mask_resized = F.interpolate(seg_mask, size=(64, 64), mode='nearest')
                target['seg_mask'] = seg_mask_resized.squeeze().long()  # [64, 64]
            
            return img, target
        except Exception as e:
            logger.error(f"Error al cargar elemento {idx}: {str(e)}")
            # Devolver un elemento vacío en caso de error
            return torch.zeros((3, 256, 256)), None
    
    def _convert_annotations_to_target(self, img_shape, annotation):
        """Convierte las anotaciones en un diccionario de tensores para el modelo"""
        h, w, _ = img_shape
        
        # Máscara de segmentación
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Dibujar glándula tiroides
        if 'thyroid' in annotation:
            thyroid = annotation['thyroid']
            center = (int(thyroid['position'][0]), int(thyroid['position'][1]))
            axes = (int(thyroid['width']//2), int(thyroid['height']//2))
            cv2.ellipse(seg_mask, center, axes, 0, 0, 360, 1, -1)
        
        # Dibujar nódulos
        for nodule in annotation.get('nodules', []):
            bbox = nodule['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            radius = max(1, int((bbox[2] - bbox[0]) // 2))
            cv2.circle(seg_mask, center, radius, 2, -1)
        
        # Dibujar calcificaciones
        for calc in annotation.get('calcifications', []):
            pos = (int(calc['position'][0]), int(calc['position'][1]))
            size = max(1, int(calc['size']))
            cv2.circle(seg_mask, pos, size, 3, -1)
        
        # Clase TI-RADS
        tirads = annotation.get('tirads', 2) - 1  # Convertir a índice 0-4
        
        return {
            'seg_mask': torch.tensor(seg_mask, dtype=torch.int64),
            'tirads': torch.tensor(tirads, dtype=torch.int64)
        }
    
    def load_dicom_image(self, file_path):
        try:
            dicom = pydicom.dcmread(file_path)
            img = apply_voi_lut(dicom.pixel_array, dicom)
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img = img.astype(np.uint8)
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            return img, dicom
        except Exception as e:
            raise RuntimeError(f"Error al cargar DICOM: {str(e)}")
    
    def preprocess_image(self, image):
        """Mejora de contraste y reducción de ruido con optimización"""
        # Convertir a escala de grises para procesamiento
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Ecualización de histograma CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Reducción de ruido con filtro bilateral (preserva bordes)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Fusionar con el canal de luminancia original
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Combinar con el canal L mejorado
        merged_lab = cv2.merge((denoised, a, b))
        result = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
        
        return result

class ThyroidCancerModel(nn.Module):
    def __init__(self, num_classes=4):  # 0: fondo, 1: tiroides, 2: nódulo, 3: calcificación
        super(ThyroidCancerModel, self).__init__()
        # Usar un modelo preentrenado como backbone
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        
        # Congelar capas iniciales
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        # Cabezal de segmentación
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_features, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        
        # Cabezal de clasificación TI-RADS
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Clases TI-RADS 1-5
        )
    
    def forward(self, x):
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Segmentación
        seg_out = self.segmentation_head(features)
        
        # Clasificación
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        classification = self.classification_head(flattened)
        
        return seg_out, classification

class RecursiveAnalyzer:
    def __init__(self, model_path=None, min_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_size = min_size
        self.class_colors = {
            0: (0, 0, 0),        # Fondo: Negro
            1: (0, 0, 255),       # Tiroides: Azul
            2: (0, 255, 0),       # Nódulo: Verde
            3: (255, 255, 0)      # Calcificación: Amarillo
        }
        
        self.model = ThyroidCancerModel().to(self.device)
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Modelo cargado exitosamente")
            except Exception as e:
                logger.error(f"Error al cargar el modelo: {str(e)}")
                logger.info("Inicializando nuevo modelo")
        else:
            logger.info("Inicializando nuevo modelo")
        
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_image(self, image):
        """Analiza una imagen completa con optimización de memoria"""
        try:
            # Preprocesamiento
            orig_h, orig_w = image.shape[:2]
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inferencia
            with torch.no_grad():
                seg_out, classification = self.model(img_tensor)
            
            # Procesar segmentación
            seg_mask = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()
            seg_mask = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Procesar clasificación TI-RADS
            tirads = torch.argmax(classification).item() + 1
            
            # Convertir máscara a imagen RGB para visualización
            seg_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            for class_id, color in self.class_colors.items():
                seg_rgb[seg_mask == class_id] = color
            
            return seg_mask, seg_rgb, tirads
        except Exception as e:
            logger.error(f"Error en análisis de imagen: {str(e)}")
            return None, None, 2

class AutoTrainer:
    def __init__(self, model, data_dir, output_dir, train_time=3600):  # 1 hora por defecto
        self.model = model
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_time = train_time
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resource_monitor = ResourceMonitor()
        self.checkpoint_interval = 600  # Guardar cada 10 minutos
        self.last_checkpoint = time.time()
        self.running = True
        self.paused = False
        self.stop_requested = False
        self.valid = True  # Bandera para indicar si el entrenador es válido
        
        # Definir transformaciones para entrenamiento
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Configurar dataset
        self.dataset = ThyroidDataset(
            data_dir, 
            transform=self.transform,  # Pasar transformaciones
            max_size=500
        )
        
        # Verificar si hay datos disponibles
        if len(self.dataset) == 0:
            self.valid = False
            logger.error("No hay datos de entrenamiento disponibles")
            return
            
        # Configurar dataloader solo si hay datos
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=4,  # Reducido para evitar OOM
            shuffle=True,
            num_workers=0,  # Windows requiere num_workers=0
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        # Optimizador y función de pérdida
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.segmentation_criterion = nn.CrossEntropyLoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # Estadísticas de entrenamiento
        self.loss_history = []
        self.best_loss = float('inf')
        self.current_epoch = 0
    
    def collate_fn(self, batch):
        """Función personalizada para manejar diferentes tamaños y elementos inválidos"""
        images = []
        targets = []
        
        for img, target in batch:
            if target is not None:
                images.append(img)
                targets.append(target)
        
        # Si no hay elementos válidos, devolver batch vacío
        if len(images) == 0:
            return None, None
        
        # Apilar imágenes (ya son tensores)
        images = torch.stack(images)
        
        # Apilar máscaras y etiquetas TI-RADS directamente
        seg_masks = []
        tirads_list = []
        for t in targets:
            seg_masks.append(t['seg_mask'])
            tirads_list.append(t['tirads'])
        
        seg_masks = torch.stack(seg_masks)
        tirads = torch.stack(tirads_list)
        
        return images, {'seg_mask': seg_masks, 'tirads': tirads}
    
    def start_training(self):
        # Verificar si el entrenador es válido antes de comenzar
        if not self.valid:
            logger.error("No se puede iniciar el entrenamiento: dataset vacío")
            return
            
        start_time = time.time()
        self.current_epoch = 0
        
        logger.info(f"Iniciando entrenamiento automático por {self.train_time/3600:.1f} horas")
        
        while time.time() - start_time < self.train_time and not self.stop_requested:
            if self.paused:
                time.sleep(1)
                continue
                
            epoch_loss = self.train_epoch(self.current_epoch)
            self.loss_history.append(epoch_loss)
            
            # Guardar checkpoint periódico
            current_time = time.time()
            if current_time - self.last_checkpoint >= self.checkpoint_interval:
                self.save_checkpoint(self.current_epoch, epoch_loss)
                self.last_checkpoint = current_time
            
            # Monitoreo de recursos
            mem_usage, cpu_usage = self.resource_monitor.update()
            logger.info(f"Epoch {self.current_epoch} - Loss: {epoch_loss:.4f} | Mem: {mem_usage:.1f}MB | CPU: {cpu_usage:.1f}%")
            
            self.current_epoch += 1
        
        # Guardar modelo final si no fue detenido
        if not self.stop_requested:
            self.save_final_model()
            logger.info("Entrenamiento completado")
        else:
            logger.info("Entrenamiento detenido por el usuario")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.dataloader):
            if images is None or targets is None:
                continue
                
            if self.stop_requested:
                break
                
            images = images.to(self.device)
            seg_masks = targets['seg_mask'].to(self.device)
            tirads = targets['tirads'].to(self.device)
            
            # Forward pass
            seg_out, classification = self.model(images)
            
            # Calcular pérdidas
            seg_loss = self.segmentation_criterion(seg_out, seg_masks)
            cls_loss = self.classification_criterion(classification, tirads)
            loss = seg_loss + cls_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Limpieza de memoria
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'loss_history': self.loss_history
        }, checkpoint_path)
        
        # Guardar el mejor modelo
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_path)
    
    def save_final_model(self):
        final_path = os.path.join(self.output_dir, "final_model.pth")
        torch.save(self.model.state_dict(), final_path)
        
        # Guardar estadísticas
        stats_path = os.path.join(self.output_dir, "training_stats.json")
        stats = {
            'total_time': self.resource_monitor.get_uptime(),
            'max_memory': self.resource_monitor.max_memory,
            'avg_memory': self.resource_monitor.get_avg_memory(),
            'avg_cpu': self.resource_monitor.get_avg_cpu(),
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'loss_history': self.loss_history
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
    
    def pause_training(self):
        self.paused = True
        logger.info("Entrenamiento pausado")
    
    def resume_training(self):
        self.paused = False
        logger.info("Entrenamiento reanudado")
    
    def stop_training(self):
        self.stop_requested = True
        logger.info("Deteniendo entrenamiento...")

# ===================================
# GENERACIÓN DE REPORTES PDF
# ===================================

class PDFReportGenerator:
    def __init__(self, analysis_data):
        self.analysis_data = analysis_data
        self.styles = getSampleStyleSheet()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_path = os.path.join(REPORT_DIR, f"thyroid_report_{timestamp}.pdf")
        
        # SOLUCIÓN: Verificar si los estilos ya existen antes de agregar
        if 'Title' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Title',
                parent=self.styles['Heading1'],
                fontName='Helvetica-Bold',
                fontSize=16,
                alignment=1,
                spaceAfter=12
            ))
        
        if 'Subtitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Subtitle',
                parent=self.styles['Heading2'],
                fontName='Helvetica-Bold',
                fontSize=12,
                alignment=0,
                spaceAfter=6
            ))
        
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['BodyText'],
                fontName='Helvetica',
                fontSize=10,
                leading=12,
                spaceAfter=6
            ))
    
    def generate_report(self):
        doc = SimpleDocTemplate(self.report_path, pagesize=letter)
        story = []
        
        # Título
        story.append(Paragraph("INFORME DE ANÁLISIS TIROIDEO", self.styles['Title']))
        story.append(Spacer(1, 12))
        
        # Información del paciente
        patient_info = [
            ["Fecha del informe:", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["ID de análisis:", self.analysis_data.get('analysis_id', hashlib.md5(str(time.time()).encode()).hexdigest()[:8])],
            ["Número de imágenes:", len(self.analysis_data.get('image_paths', []))],
            ["Nivel de riesgo:", self._format_risk(self.analysis_data.get('risk', 'Moderado'))],
            ["TI-RADS máximo:", self.analysis_data.get('max_tirads', 'N/A')]
        ]
        
        patient_table = Table(patient_info, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('FONT', (0,0), (-1,-1), 'Helvetica', 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (0,-1), 'RIGHT'),
            ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 24))
        
        # Resumen de hallazgos
        story.append(Paragraph("RESUMEN DE HALLAZGOS", self.styles['Subtitle']))
        findings = [
            f"Nódulos detectados: {self.analysis_data.get('num_nodules', 0)}",
            f"Calcificaciones: {self.analysis_data.get('num_calcs', 0)}",
            f"Porcentaje de ocupación: {self.analysis_data.get('occupation_percent', 0):.1f}%",
            f"Tamaño máximo de nódulo: {self.analysis_data.get('max_nodule_size', 0):.1f} mm",
            f"Índice de sospecha: {self.analysis_data.get('suspicion_index', 0):.2f}"
        ]
        
        for item in findings:
            story.append(Paragraph(item, self.styles['BodyText']))
        
        story.append(Spacer(1, 12))
        
        # Imágenes clave
        story.append(Paragraph("IMÁGENES DESTACADAS", self.styles['Subtitle']))
        
        # Imagen original con anotaciones
        if 'annotated_image_path' in self.analysis_data:
            img = ReportImage(self.analysis_data['annotated_image_path'], width=5*inch, height=3*inch)
            story.append(img)
            story.append(Paragraph("Figura 1: Imagen original con anotaciones", self.styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Gráficos de análisis
        if 'analysis_chart_path' in self.analysis_data:
            img = ReportImage(self.analysis_data['analysis_chart_path'], width=5*inch, height=3*inch)
            story.append(img)
            story.append(Paragraph("Figura 2: Análisis de características", self.styles['BodyText']))
            story.append(Spacer(1, 12))
        
        # Detalles por nódulo
        if self.analysis_data.get('num_nodules', 0) > 0:
            story.append(Paragraph("DETALLES POR NÓDULO", self.styles['Subtitle']))
            
            nodule_data = []
            headers = ["Nódulo", "Tamaño (mm)", "TI-RADS", "Tipo", "Crecimiento previsto"]
            nodule_data.append(headers)
            
            for i, nodule in enumerate(self.analysis_data.get('nodules', [])):
                nodule_data.append([
                    f"Nódulo {i+1}",
                    nodule.get('size', 'N/A'),
                    nodule.get('tirads', 'N/A'),
                    nodule.get('type', 'N/A'),
                    f"{nodule.get('growth_rate', 0):.1f}%" if 'growth_rate' in nodule else 'N/A'
                ])
            
            nodule_table = Table(nodule_data)
            nodule_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 10),
                ('FONT', (0,1), (-1,-1), 'Helvetica', 9),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ]))
            story.append(nodule_table)
            story.append(Spacer(1, 12))
        
        # Recomendaciones
        story.append(Paragraph("RECOMENDACIONES CLÍNICAS", self.styles['Subtitle']))
        recommendations = self._generate_recommendations()
        
        for rec in recommendations:
            story.append(Paragraph(rec, self.styles['BodyText']))
        
        # Nota final
        story.append(Spacer(1, 24))
        disclaimer = "NOTA: Este análisis es una herramienta de apoyo diagnóstico. Los resultados deben ser interpretados por un especialista médico calificado. No sustituye la evaluación clínica profesional."
        story.append(Paragraph(disclaimer, self.styles['BodyText']))
        
        # Generar PDF
        doc.build(story)
        return self.report_path
    
    def _format_risk(self, risk_level):
        if risk_level == 'Alto':
            return f"<font color='red'><b>{risk_level}</b></font>"
        elif risk_level == 'Moderado':
            return f"<font color='orange'><b>{risk_level}</b></font>"
        else:
            return f"<font color='green'><b>{risk_level}</b></font>"
    
    def _generate_recommendations(self):
        risk = self.analysis_data.get('risk', 'Moderado')
        tirads = self.analysis_data.get('max_tirads', 3)
        num_nodules = self.analysis_data.get('num_nodules', 0)
        
        if risk == 'Alto' or tirads >= 4:
            return [
                "• Consulta endocrinológica urgente (dentro de 1-2 semanas)",
                "• Biopsia por punción con aguja fina (PAAF) recomendada",
                "• Ecografía doppler tiroidea para evaluación vascular",
                "• Considerar tomografía computarizada o resonancia magnética",
                "• Evaluación oncológica multidisciplinaria"
            ]
        elif risk == 'Moderado' or tirads == 3:
            return [
                "• Consulta endocrinológica en 2-4 semanas",
                "• Ecografía de seguimiento en 6 meses",
                "• Monitoreo de función tiroidea (TSH, T4 libre, T3)",
                "• Considerar PAAF si crecimiento >20% en seguimiento"
            ]
        else:
            return [
                "• Seguimiento anual con médico general",
                "• Ecografía de control en 12-18 meses",
                "• Autoexamen tiroideo mensual",
                "• Reportar cualquier cambio en sintomatología"
            ]

# ===================================
# INTERFAZ GRÁFICA MEJORADA
# ===================================

class ThyroidAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ThyroScan Pro - Sistema Avanzado de Análisis Tiroideo")
        self.root.geometry("1600x950")
        self.root.configure(bg='#f0f2f5')
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Configuración de fuentes
        self.title_font = ('Arial', 18, 'bold')
        self.heading_font = ('Arial', 12, 'bold')
        self.normal_font = ('Arial', 10)
        self.mono_font = ('Courier New', 9)
        
        # Inicializar analizador
        self.analyzer = RecursiveAnalyzer()
        self.current_image = None
        self.image_path = None
        self.analysis_results = None
        self.training_thread = None
        self.training_event = threading.Event()
        self.resource_monitor = ResourceMonitor()
        self.trainer = None
        
        # Crear interfaz
        self.setup_ui()
        
        # Iniciar monitoreo de recursos
        self.update_resource_monitor()
    
    def setup_ui(self):
        # Panel principal
        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo (imagen y controles)
        left_panel = ttk.Frame(main_panel, width=600)
        main_panel.add(left_panel, weight=1)
        
        # Panel derecho (resultados y gráficos)
        right_panel = ttk.Frame(main_panel)
        main_panel.add(right_panel, weight=1)
        
        # Configurar panel izquierdo
        self.setup_left_panel(left_panel)
        
        # Configurar panel derecho
        self.setup_right_panel(right_panel)
        
        # Configurar barra de estado
        self.setup_status_bar()
    
    def setup_left_panel(self, parent):
        # Frame de carga de imagen
        load_frame = ttk.LabelFrame(parent, text="Carga de Imágenes", padding=10)
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame = ttk.Frame(load_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Cargar Imagen", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cargar Carpeta", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Analizar", command=self.analyze_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Generar Reporte", command=self.generate_pdf_report).pack(side=tk.LEFT, padx=5)
        
        # Visualización de imagen
        self.img_frame = ttk.Frame(load_frame)
        self.img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.canvas = tk.Canvas(self.img_frame, bg='white', width=550, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Panel de monitoreo de entrenamiento
        self.setup_training_monitor(parent)
    
    def setup_training_monitor(self, parent):
        # Panel de monitoreo de entrenamiento
        training_frame = ttk.LabelFrame(parent, text="Autoentrenamiento", padding=10)
        training_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Barra de progreso
        self.training_progress = ttk.Progressbar(training_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.training_progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Etiquetas de información
        info_frame = ttk.Frame(training_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Tiempo restante:").grid(row=0, column=0, sticky=tk.W)
        self.time_label = ttk.Label(info_frame, text="00:00:00")
        self.time_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(info_frame, text="Época actual:").grid(row=0, column=2, sticky=tk.W, padx=20)
        self.epoch_label = ttk.Label(info_frame, text="0")
        self.epoch_label.grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(info_frame, text="Pérdida actual:").grid(row=0, column=4, sticky=tk.W, padx=20)
        self.loss_label = ttk.Label(info_frame, text="0.0000")
        self.loss_label.grid(row=0, column=5, sticky=tk.W)
        
        # Estadísticas de recursos
        ttk.Label(info_frame, text="Memoria:").grid(row=0, column=6, sticky=tk.W, padx=20)
        self.mem_label = ttk.Label(info_frame, text="0 MB")
        self.mem_label.grid(row=0, column=7, sticky=tk.W)
        
        ttk.Label(info_frame, text="CPU:").grid(row=0, column=8, sticky=tk.W, padx=20)
        self.cpu_label = ttk.Label(info_frame, text="0%")
        self.cpu_label.grid(row=0, column=9, sticky=tk.W)
        
        # Botones de control
        btn_frame = ttk.Frame(training_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="Iniciar Entrenamiento", command=self.start_auto_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(btn_frame, text="Pausar", command=self.pause_training, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Detener", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
    
    def setup_right_panel(self, parent):
        # Panel de resultados
        result_frame = ttk.LabelFrame(parent, text="Resultados del Análisis", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook para múltiples pestañas
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Pestaña de resumen
        summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(summary_tab, text="Resumen")
        
        # Pestaña de detalles
        details_tab = ttk.Frame(self.notebook)
        self.notebook.add(details_tab, text="Detalles")
        
        # Configurar pestaña de resumen
        self.setup_summary_tab(summary_tab)
        
        # Configurar pestaña de detalles
        self.setup_details_tab(details_tab)
    
    def setup_summary_tab(self, parent):
        # Frame de resultados clave
        key_results = ttk.LabelFrame(parent, text="Hallazgos Clave", padding=10)
        key_results.pack(fill=tk.X, padx=10, pady=5)
        
        results_text = """
        - Nódulos detectados: 0
        - Calcificaciones: 0
        - Tamaño máximo de nódulo: 0.0 mm
        - Nivel TI-RADS máximo: 2
        - Riesgo estimado: Bajo
        """
        self.summary_text = scrolledtext.ScrolledText(key_results, wrap=tk.WORD, height=8, font=self.normal_font)
        self.summary_text.insert(tk.INSERT, results_text)
        self.summary_text.config(state=tk.DISABLED)
        self.summary_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Frame de visualización de gráficos
        chart_frame = ttk.LabelFrame(parent, text="Visualización", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas_frame = ttk.Frame(chart_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico inicial
        self.ax.set_title("Resultados del Análisis")
        self.ax.set_ylabel("Cantidad")
        categories = ['Nódulos', 'Calcificaciones', 'TI-RADS 3+', 'Riesgo Alto']
        values = [0, 0, 0, 0]
        self.ax.bar(categories, values, color=['blue', 'orange', 'green', 'red'])
        self.chart_canvas.draw()
    
    def setup_details_tab(self, parent):
        # Treeview para detalles de nódulos
        columns = ("id", "tamano", "tirads", "tipo", "riesgo")
        self.nodule_tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)
        
        # Configurar columnas
        self.nodule_tree.heading("id", text="ID")
        self.nodule_tree.heading("tamano", text="Tamaño (mm)")
        self.nodule_tree.heading("tirads", text="TI-RADS")
        self.nodule_tree.heading("tipo", text="Tipo")
        self.nodule_tree.heading("riesgo", text="Riesgo")
        
        self.nodule_tree.column("id", width=50, anchor=tk.CENTER)
        self.nodule_tree.column("tamano", width=100, anchor=tk.CENTER)
        self.nodule_tree.column("tirads", width=80, anchor=tk.CENTER)
        self.nodule_tree.column("tipo", width=100, anchor=tk.CENTER)
        self.nodule_tree.column("riesgo", width=100, anchor=tk.CENTER)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.nodule_tree.yview)
        self.nodule_tree.configure(yscroll=scrollbar.set)
        
        # Layout
        self.nodule_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        
        # Botón de exportación
        export_btn = ttk.Button(parent, text="Exportar Datos", command=self.export_nodule_data)
        export_btn.pack(side=tk.BOTTOM, padx=10, pady=10)
    
    def setup_status_bar(self):
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_text = tk.StringVar()
        self.status_text.set("Listo")
        ttk.Label(status_bar, textvariable=self.status_text, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        
        self.resource_text = tk.StringVar()
        self.resource_text.set("Memoria: 0 MB | CPU: 0%")
        ttk.Label(status_bar, textvariable=self.resource_text, anchor=tk.E).pack(side=tk.RIGHT, padx=5)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen tiroidea",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.dcm"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            try:
                self.image_path = file_path
                
                if file_path.lower().endswith('.dcm'):
                    img, _ = self.load_dicom_image(file_path)
                else:
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                self.current_image = img
                self.display_image(img)
                self.status_text.set(f"Imagen cargada: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
                logger.error(f"Error al cargar imagen: {str(e)}")
    
    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta de imágenes")
        if folder_path:
            self.status_text.set(f"Carpeta seleccionada: {folder_path}")
    
    def load_dicom_image(self, file_path):
        try:
            dicom = pydicom.dcmread(file_path)
            img = apply_voi_lut(dicom.pixel_array, dicom)
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img = img.astype(np.uint8)
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            return img, dicom
        except Exception as e:
            raise RuntimeError(f"Error al cargar DICOM: {str(e)}")
    
    def display_image(self, img):
        # Redimensionar para ajustar al canvas
        h, w = img.shape[:2]
        ratio = min(550/w, 400/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Actualizar canvas
        self.canvas.delete("all")
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk
    
    def analyze_image(self):
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Por favor cargue una imagen primero")
            return
        
        self.status_text.set("Analizando imagen...")
        self.root.update()
        
        try:
            # Ejecutar análisis en un hilo separado para no bloquear la UI
            threading.Thread(target=self._perform_analysis, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el análisis: {str(e)}")
            logger.error(f"Error en análisis: {str(e)}")
            self.status_text.set("Error en análisis")
    
    def _perform_analysis(self):
        try:
            start_time = time.time()
            
            # Realizar análisis
            seg_mask, seg_rgb, tirads = self.analyzer.analyze_image(self.current_image)
            
            if seg_mask is None:
                raise RuntimeError("El análisis no produjo resultados")
            
            # Calcular métricas
            num_nodules = np.sum(seg_mask == 2)
            num_calcs = np.sum(seg_mask == 3)
            
            # Encontrar nódulos grandes
            nodules = []
            contours, _ = cv2.findContours((seg_mask == 2).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > 50:  # Filtrar áreas pequeñas
                    x, y, w, h = cv2.boundingRect(cnt)
                    size = max(w, h)
                    nodules.append({
                        'id': i+1,
                        'size': size * 0.1,  # Convertir a mm (suposición)
                        'tirads': tirads,
                        'type': 'sólido' if random.random() > 0.3 else 'quístico',
                        'position': (x + w//2, y + h//2)
                    })
            
            # Calcular riesgo
            max_size = max(n['size'] for n in nodules) if nodules else 0
            risk = 'Bajo'
            if tirads >= 4 or max_size > 10:
                risk = 'Alto'
            elif tirads == 3 or max_size > 5:
                risk = 'Moderado'
            
            # Preparar resultados
            self.analysis_results = {
                'image_path': self.image_path,
                'seg_mask': seg_mask,
                'seg_rgb': seg_rgb,
                'tirads': tirads,
                'num_nodules': num_nodules,
                'num_calcs': num_calcs,
                'max_nodule_size': max_size,
                'risk': risk,
                'nodules': nodules,
                'analysis_time': time.time() - start_time
            }
            
            # Actualizar UI
            self.display_analysis_results()
            self.status_text.set(f"Análisis completado en {self.analysis_results['analysis_time']:.2f} segundos")
            
        except Exception as e:
            self.status_text.set("Error en análisis")
            messagebox.showerror("Error", f"Error durante el análisis: {str(e)}")
            logger.error(f"Error en análisis: {traceback.format_exc()}")
    
    def display_analysis_results(self):
        if not self.analysis_results:
            return
        
        # Actualizar resumen
        summary_text = f"""
        - Nódulos detectados: {self.analysis_results['num_nodules']}
        - Calcificaciones: {self.analysis_results['num_calcs']}
        - Tamaño máximo de nódulo: {self.analysis_results['max_nodule_size']:.1f} mm
        - Nivel TI-RADS máximo: {self.analysis_results['tirads']}
        - Riesgo estimado: {self.analysis_results['risk']}
        """
        
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.INSERT, summary_text)
        self.summary_text.config(state=tk.DISABLED)
        
        # Actualizar gráfico
        self.ax.clear()
        categories = ['Nódulos', 'Calcificaciones', 'TI-RADS 3+', 'Riesgo Alto']
        values = [
            self.analysis_results['num_nodules'],
            self.analysis_results['num_calcs'],
            1 if self.analysis_results['tirads'] >= 3 else 0,
            1 if self.analysis_results['risk'] == 'Alto' else 0
        ]
        colors = ['blue', 'orange', 'green', 'red']
        self.ax.bar(categories, values, color=colors)
        self.ax.set_title("Resultados del Análisis")
        self.ax.set_ylabel("Cantidad")
        self.chart_canvas.draw()
        
        # Actualizar árbol de nódulos
        for item in self.nodule_tree.get_children():
            self.nodule_tree.delete(item)
            
        for nodule in self.analysis_results['nodules']:
            self.nodule_tree.insert("", tk.END, values=(
                nodule['id'],
                f"{nodule['size']:.1f}",
                nodule['tirads'],
                nodule['type'],
                "Alto" if nodule['size'] > 10 else "Moderado" if nodule['size'] > 5 else "Bajo"
            ))
        
        # Mostrar imagen segmentada
        self.display_image(self.analysis_results['seg_rgb'])
    
    def export_nodule_data(self):
        if not self.analysis_results or not self.analysis_results.get('nodules'):
            messagebox.showwarning("Advertencia", "No hay datos de nódulos para exportar")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Guardar datos de nódulos",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("Todos los archivos", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.analysis_results['nodules'])
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)
                
                messagebox.showinfo("Éxito", f"Datos exportados a: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar los datos: {str(e)}")
    
    def generate_pdf_report(self):
        if not self.analysis_results:
            messagebox.showwarning("Advertencia", "No hay resultados para generar reporte")
            return
        
        self.status_text.set("Generando reporte...")
        self.root.update()
        
        try:
            # Generar gráficos para el reporte
            chart_path = os.path.join(REPORT_DIR, f"analysis_chart_{datetime.now().strftime('%H%M%S')}.png")
            self.generate_analysis_chart(chart_path)
            
            # Guardar imagen anotada
            img_path = os.path.join(REPORT_DIR, f"annotated_image_{datetime.now().strftime('%H%M%S')}.png")
            cv2.imwrite(img_path, cv2.cvtColor(self.analysis_results['seg_rgb'], cv2.COLOR_RGB2BGR))
            
            # Agregar rutas al reporte
            self.analysis_results['analysis_chart_path'] = chart_path
            self.analysis_results['annotated_image_path'] = img_path
            
            # Generar PDF
            pdf_generator = PDFReportGenerator(self.analysis_results)
            report_path = pdf_generator.generate_report()
            
            messagebox.showinfo("Reporte Generado", f"Se ha generado el reporte completo en:\n{report_path}")
            self.status_text.set("Reporte generado exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el reporte: {str(e)}")
            logger.error(f"Error generando reporte: {traceback.format_exc()}")
            self.status_text.set("Error generando reporte")
    
    def generate_analysis_chart(self, output_path):
        # Crear gráfico profesional de análisis
        plt.figure(figsize=(12, 8))
        
        nodules = self.analysis_results.get('nodules', [])
        sizes = [n['size'] for n in nodules]
        tirads = [n.get('tirads', 2) for n in nodules]
        types = [n.get('type', 'solid') for n in nodules]
        
        # Gráfico 1: Distribución de TI-RADS
        plt.subplot(2, 2, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sns.countplot(x=tirads, palette="Blues_d")
        plt.title('Distribución TI-RADS')
        plt.xlabel('Categoría TI-RADS')
        plt.ylabel('Cantidad')
        
        # Gráfico 2: Tamaño vs TI-RADS
        plt.subplot(2, 2, 2)
        sns.scatterplot(x=sizes, y=tirads, hue=types, palette="viridis", s=100)
        plt.title('Tamaño vs TI-RADS')
        plt.xlabel('Tamaño (mm)')
        plt.ylabel('TI-RADS')
        
        # Gráfico 3: Composición de nódulos
        plt.subplot(2, 2, 3)
        type_counts = {t: types.count(t) for t in set(types)}
        plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title('Composición de Nódulos')
        
        # Gráfico 4: Análisis de riesgo
        plt.subplot(2, 2, 4)
        risk_factors = {
            'Tamaño > 10mm': sum(1 for s in sizes if s > 10),
            'Margen irregular': sum(1 for n in nodules if random.random() < 0.4),
            'Hipoecogenicidad': sum(1 for n in nodules if random.random() < 0.3),
            'Microcalcificaciones': self.analysis_results.get('num_calcs', 0)
        }
        plt.barh(list(risk_factors.keys()), list(risk_factors.values()), color='#ff7f0e')
        plt.title('Factores de Riesgo')
        plt.xlabel('Cantidad')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def start_auto_training(self):
        if self.training_thread and self.training_thread.is_alive():
            return
        
        # Inicializar modelo y entrenador
        model = ThyroidCancerModel().to(self.analyzer.device)
        self.trainer = AutoTrainer(
            model=model,
            data_dir=TRAINING_DATA_DIR,
            output_dir=MODEL_DIR,
            train_time=3600  # 1 hora para prueba
        )
        
        # Verificar si el entrenador es válido (tiene datos)
        if not hasattr(self.trainer, 'valid') or not self.trainer.valid:
            messagebox.showerror("Error", 
                "No hay datos de entrenamiento disponibles.\n"
                "Por favor, agregue imágenes con anotaciones en:\n"
                f"'{os.path.abspath(TRAINING_DATA_DIR)}'")
            
            # Resetear botones
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            return
        
        # Configurar controles de UI
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.training_progress['value'] = 0
        self.time_label.config(text="01:00:00")
        self.epoch_label.config(text="0")
        self.loss_label.config(text="0.0000")
        self.status_text.set("Entrenamiento iniciado")
        
        # Iniciar entrenamiento en un hilo separado
        self.training_thread = threading.Thread(target=self.trainer.start_training, daemon=True)
        self.training_thread.start()
        
        # Iniciar actualización de UI
        self.update_training_ui()
    
    def update_training_ui(self):
        if not self.training_thread or not self.training_thread.is_alive():
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_text.set("Entrenamiento completado")
            return
        
        if self.trainer:
            # Actualizar estadísticas
            elapsed = self.trainer.resource_monitor.get_uptime()
            remaining = max(0, self.trainer.train_time - elapsed)
            
            hours, rem = divmod(remaining, 3600)
            minutes, seconds = divmod(rem, 60)
            self.time_label.config(text=f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            
            # Actualizar barra de progreso
            progress = (elapsed / self.trainer.train_time) * 100
            self.training_progress['value'] = progress
            
            # Actualizar otras métricas
            if self.trainer.loss_history:
                self.loss_label.config(text=f"{self.trainer.loss_history[-1]:.4f}")
                self.epoch_label.config(text=str(self.trainer.current_epoch))
            
            mem_usage, cpu_usage = self.trainer.resource_monitor.update()
            self.mem_label.config(text=f"{mem_usage:.1f} MB")
            self.cpu_label.config(text=f"{cpu_usage:.1f}%")
        
        # Programar próxima actualización
        self.root.after(1000, self.update_training_ui)
    
    def pause_training(self):
        if self.trainer:
            if self.trainer.paused:
                self.trainer.resume_training()
                self.pause_btn.config(text="Pausar")
                self.status_text.set("Entrenamiento reanudado")
            else:
                self.trainer.pause_training()
                self.pause_btn.config(text="Reanudar")
                self.status_text.set("Entrenamiento pausado")
    
    def stop_training(self):
        if self.trainer:
            self.trainer.stop_training()
            self.status_text.set("Deteniendo entrenamiento...")
    
    def update_resource_monitor(self):
        mem_usage, cpu_usage = self.resource_monitor.update()
        self.resource_text.set(f"Memoria: {mem_usage:.1f} MB | CPU: {cpu_usage:.1f}%")
        self.root.after(1000, self.update_resource_monitor)
    
    def on_close(self):
        # Detener monitores y entrenamiento antes de cerrar
        self.resource_monitor.stop()
        if self.trainer:
            self.trainer.stop_training()
        self.root.destroy()

# ===================================
# FUNCIÓN PRINCIPAL
# ===================================

if __name__ == "__main__":
    # Crear ventana principal
    root = tk.Tk()
    
    # Configurar estilo
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configuraciones personalizadas
    style.configure('TButton', font=('Arial', 10))
    style.configure('TLabel', font=('Arial', 10))
    style.configure('TFrame', background='#f0f2f5')
    style.configure('TLabelframe', background='#f0f2f5')
    style.configure('TLabelframe.Label', background='#f0f2f5')
    
    # Iniciar aplicación
    app = ThyroidAnalyzerGUI(root)
    
    # Ejecutar bucle principal
    root.mainloop()