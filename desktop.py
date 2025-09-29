import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import shutil
import warnings
import collections
import math
from typing import Tuple
from model_main import *
import torch.nn.functional as F
import datetime
warnings.filterwarnings("ignore")

class ImageClassiferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shrimp Disease Classifier")
        self.root.geometry("800x800")
        self.root.configure(bg='#808080')  # Gray background
        self.current_image_index = 0
        self.image_paths = []
        self.predictions = []
        self.model = None
        self.class_names = ["BG", "Healthy", "WSSV", "Yellowhead"]
        self.load_model()
        self.create_widgets()
        self.add_author_credit()
        self.output_base_dir = "classified_images"
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
            for class_name in self.class_names:
                os.makedirs(os.path.join(self.output_base_dir, class_name))    
    def create_model_architecture(self):
        return feathernetx_tiny(num_classes = len(self.class_names))    
    
    def load_model(self):
        model_data = torch.load('best_model.pth', map_location=torch.device('cpu'))
        self.model = self.create_model_architecture()
        self.model.load_state_dict(model_data)
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def add_author_credit(self):
        credit_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        credit_frame.pack(side=tk.BOTTOM, fill=tk.X)
        credit_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(credit_frame, text="Development Team:", font=('Arial', 9, 'bold'), 
                            bg='#2c3e50', fg='white')
        title_label.pack()
        
        # Create a canvas for scrolling if you have many authors
        canvas = tk.Canvas(credit_frame, bg='#2c3e50', highlightthickness=0, height=20)
        scrollbar = ttk.Scrollbar(credit_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg='#2c3e50')
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)
        
        # List of authors
        authors = ["Ms. Sandhya Sharma",  "Dr. Shinya Watanabe", 
                "Dr. Satoshi Kondo", "Dr. Bishnu Prasad Gautam", "Dr. Kazuhiko Sato"]
        
        # Add authors to scrollable frame
        for i, author in enumerate(authors):
            label = tk.Label(scrollable_frame, text=author, font=('Arial', 11), 
                            bg='#2c3e50', fg='white', padx=5)
            label.grid(row=0, column=i, sticky="ew")
        
        canvas.pack(side="top", fill="x", expand=True)
        scrollbar.pack(side="bottom", fill="x")
        
        # Copyright year - centered at the bottom
        copyright_label = tk.Label(credit_frame, text=f"© {datetime.datetime.now().year}", 
                                font=('Arial', 12), bg='#2c3e50', fg='white')
        copyright_label.pack(side=tk.BOTTOM, pady=5)

    def create_widgets(self):
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=100)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        title_frame.pack_propagate(False)  
        try:
            left_logo_image = Image.open("logo.png")
            left_logo_image = left_logo_image.resize((200, 70), Image.LANCZOS)
            self.left_logo_photo = ImageTk.PhotoImage(left_logo_image)
            left_logo_label = tk.Label(title_frame, image=self.left_logo_photo, bg='#ffcc99')
            left_logo_label.pack(side=tk.LEFT, padx=(20, 15), pady=15)
        except:
            left_logo_label = tk.Label(title_frame, text="LOGO", bg='#ffcc99', width=20, height=3)
            left_logo_label.pack(side=tk.LEFT, padx=(20, 15), pady=15)
        
        title_label = tk.Label(title_frame, text="SHRIMP DISEASE CLASSIFIER", font=('Arial', 28, 'bold'), bg='#2c3e50', fg='white', pady=10)
        title_label.pack(expand=True)
        button_frame = tk.Frame(self.root, bg='#808080')
        button_frame.pack(pady=20)
        button_style = {'bg': 'black', 'fg': 'white', 'font': ('Arial', 13, 'bold'), 'width': 15, 'height': 2,'bd': 0, 'relief': 'flat','activebackground': '#333333','activeforeground': 'white'}
        tk.Button(button_frame, text="Import Image", command=self.import_image, **button_style).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(button_frame, text="Import Folder", command=self.import_folder, **button_style).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(button_frame, text="Next", command=self.next_image, **button_style).pack(side=tk.LEFT, padx=10, pady=5)

        button_frame2 = tk.Frame(self.root, bg='#808080')
        button_frame2.pack(pady=10)
        
        tk.Button(button_frame2, text="Delete", command=self.delete_image, **button_style).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(button_frame2, text="Retrieve", command=self.retrieve_image, **button_style).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(button_frame2, text="Save", command=self.save_image, **button_style).pack(side=tk.LEFT, padx=10, pady=5)

        self.image_frame = tk.Frame(self.root, bg='#808080')
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.canvas = tk.Canvas(self.image_frame, bg='#808080', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.prediction_label = tk.Label(self.root, text="No image loaded", font=('Arial', 14), bg='#808080', fg='white')
        self.prediction_label.pack(pady=10)

        self.probability_label = tk.Label(self.root, text="", font=('Arial', 12, 'bold'), bg='#808080', fg='black')
        self.probability_label.pack(pady=5)
    
    def import_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image_paths = [file_path]
            self.current_image_index = 0
            self.display_image(file_path)
    
    def import_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in extensions]
            
            if self.image_paths:
                self.current_image_index = 0
                self.display_image(self.image_paths[0])
            else:
                messagebox.showwarning("No Images", "No supported image files found in the selected folder.")
    
    def display_image(self, image_path):
        try:
            image = Image.open(image_path)
            self.canvas.delete("all")
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 400
            img_width, img_height = image.size
            img_ratio = img_width / img_height
            canvas_ratio = canvas_width / canvas_height
            if img_ratio > canvas_ratio:
                display_width = canvas_width
                display_height = int(canvas_width / img_ratio)
            else:
                display_height = canvas_height
                display_width = int(canvas_height * img_ratio)
            resized_image = image.resize((display_width, display_height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_image)
            x_pos = (canvas_width - display_width) // 2
            y_pos = (canvas_height - display_height) // 2
            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_image)
            self.predict_image(image_path)            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def predict_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                prediction_idx = predicted.item()
                confidence_value = confidence.item() * 100
            if 0 <= prediction_idx < len(self.class_names):
                predicted_class = self.class_names[prediction_idx]
                self.prediction_label.config(text=f"Prediction: {predicted_class}", fg='#FFFFFF', font=('Arial', 14, 'bold'))
                self.probability_label.config(text=f"Confidence: {confidence_value:.2f}%", fg='#FDD835', font=('Arial', 14, 'bold'))
                self.current_prediction = (predicted_class, confidence_value)
            else:
                self.prediction_label.config(text="Prediction: Unknown")
                self.probability_label.config(text="")                
        except Exception as e:
            self.prediction_label.config(text="Error in prediction")
            self.probability_label.config(text="")
            print(f"Prediction error: {e}")
    
    def next_image(self):
        if self.image_paths and len(self.image_paths) > 1:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.display_image(self.image_paths[self.current_image_index])
    
    def delete_image(self):
        if self.image_paths:
            del self.image_paths[self.current_image_index]            
            if self.image_paths:
                if self.current_image_index >= len(self.image_paths):
                    self.current_image_index = len(self.image_paths) - 1
                self.display_image(self.image_paths[self.current_image_index])
            else:
                self.canvas.delete("all")
                self.prediction_label.config(text="No image loaded")
                self.probability_label.config(text="")
    
    def retrieve_image(self):
        messagebox.showinfo("Info", "Retrieve functionality would be implemented here")
    
    def save_image(self):
        if self.image_paths and hasattr(self, 'current_prediction'):
            current_path = self.image_paths[self.current_image_index]
            predicted_class, confidence = self.current_prediction
            class_dir = os.path.join(self.output_base_dir, predicted_class)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            filename = os.path.basename(current_path)
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(class_dir, f"{name}_{confidence:.2f}{ext}")
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(class_dir, f"{name}_{confidence:.2f}_{counter}{ext}")
                counter += 1
            
            try:
                shutil.copy2(current_path, dest_path)
                messagebox.showinfo("Success", f"Image saved successfully in {predicted_class} folder")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No prediction available to save the image")
