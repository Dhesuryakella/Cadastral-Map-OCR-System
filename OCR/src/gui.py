import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import json
import threading
from detect import CadastralMapExtractor
import logging
import traceback

class CadastralMapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cadastral Map Analysis System")
        self.setup_logging()
        self.setup_ui()
        self.initialize_detector()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gui.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_ui(self):
        """Setup the GUI interface"""
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="5")
        file_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(file_frame, text="Map File:").grid(row=0, column=0, sticky=tk.W)
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Process button and progress bar
        control_frame = ttk.Frame(self.main_frame)
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.process_btn = ttk.Button(control_frame, text="Process Map", command=self.process_map)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Create notebook for tabbed results
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.main_frame.rowconfigure(2, weight=1)
        
        # Tab 1: Place Names
        self.place_names_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.place_names_frame, text='Place Names')
        
        # Place names tree view
        self.place_names_tree = ttk.Treeview(self.place_names_frame, columns=('Name', 'Confidence', 'Location'), show='headings')
        self.place_names_tree.heading('Name', text='Place Name')
        self.place_names_tree.heading('Confidence', text='Confidence')
        self.place_names_tree.heading('Location', text='Location')
        self.place_names_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar for place names
        place_scrollbar = ttk.Scrollbar(self.place_names_frame, orient=tk.VERTICAL, command=self.place_names_tree.yview)
        place_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.place_names_tree.configure(yscrollcommand=place_scrollbar.set)
        
        # Tab 2: Survey Numbers
        self.survey_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.survey_frame, text='Survey Numbers')
        
        # Survey numbers tree view
        self.survey_tree = ttk.Treeview(self.survey_frame, columns=('Number', 'Confidence', 'Location'), show='headings')
        self.survey_tree.heading('Number', text='Survey Number')
        self.survey_tree.heading('Confidence', text='Confidence')
        self.survey_tree.heading('Location', text='Location')
        self.survey_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar for survey numbers
        survey_scrollbar = ttk.Scrollbar(self.survey_frame, orient=tk.VERTICAL, command=self.survey_tree.yview)
        survey_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.survey_tree.configure(yscrollcommand=survey_scrollbar.set)
        
        # Tab 3: Map Features
        self.features_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.features_frame, text='Map Features')
        
        # Features tree view
        self.features_tree = ttk.Treeview(self.features_frame, columns=('Category', 'Type', 'Details'), show='headings')
        self.features_tree.heading('Category', text='Category')
        self.features_tree.heading('Type', text='Type')
        self.features_tree.heading('Details', text='Details')
        self.features_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar for features
        features_scrollbar = ttk.Scrollbar(self.features_frame, orient=tk.VERTICAL, command=self.features_tree.yview)
        features_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.features_tree.configure(yscrollcommand=features_scrollbar.set)
        
        # Configure frame weights
        self.place_names_frame.columnconfigure(0, weight=1)
        self.place_names_frame.rowconfigure(0, weight=1)
        self.survey_frame.columnconfigure(0, weight=1)
        self.survey_frame.rowconfigure(0, weight=1)
        self.features_frame.columnconfigure(0, weight=1)
        self.features_frame.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
    def initialize_detector(self):
        """Initialize the detector in a separate thread"""
        self.status_var.set("Initializing detector...")
        self.progress.start()
        
        def init_detector():
            try:
                self.detector = CadastralMapExtractor()
                self.root.after(0, self.initialization_complete)
            except Exception as e:
                self.root.after(0, lambda: self.show_error(str(e)))
                
        threading.Thread(target=init_detector).start()
        
    def initialization_complete(self):
        """Called when detector initialization is complete"""
        self.progress.stop()
        self.status_var.set("Ready")
        self.logger.info("Detector initialized successfully")
        
    def browse_file(self):
        """Open file dialog to select input map"""
        filename = filedialog.askopenfilename(
            title="Select Map",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if filename:
            self.file_path.set(filename)
            self.logger.info(f"Selected file: {filename}")
            
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        return obj

    def process_map(self):
        """Process the selected map"""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select an input map")
            return
            
        # Clear previous results
        self.place_names_tree.delete(*self.place_names_tree.get_children())
        self.survey_tree.delete(*self.survey_tree.get_children())
        self.features_tree.delete(*self.features_tree.get_children())
        
        self.process_btn.state(['disabled'])
        self.progress.start()
        self.status_var.set("Processing map...")
        
        def process():
            try:
                # Process image
                characters, numbers, all_results, symbols = self.detector.extract_map_text(self.file_path.get())
                
                # Convert numpy types before JSON serialization
                characters = self.convert_numpy_types(characters)
                numbers = self.convert_numpy_types(numbers)
                symbols = self.convert_numpy_types(symbols)
                
                # Create output directory
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                output_base = output_dir / Path(self.file_path.get()).stem
                
                # Save JSON results
                with open(f"{output_base}_results.json", 'w') as f:
                    json.dump({
                        'characters': characters,
                        'numbers': numbers,
                        'symbols': symbols
                    }, f, indent=2)
                
                # Update GUI with results
                self.root.after(0, lambda: self.show_results(characters, numbers, symbols))
                
            except Exception as e:
                error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
                self.root.after(0, lambda: self.show_error(error_msg))
                
        threading.Thread(target=process).start()
        
    def show_results(self, characters, numbers, symbols):
        """Display processing results in the GUI"""
        try:
            self.progress.stop()
            self.process_btn.state(['!disabled'])
            
            # Display place names
            for char in characters:
                confidence = char.get('confidence', 0)
                if isinstance(confidence, (int, float)):
                    confidence = f"{float(confidence):.2%}"
                location = char.get('location', (0, 0))
                if isinstance(location, (list, tuple)) and len(location) == 2:
                    location = f"({location[0]}, {location[1]})"
                else:
                    location = "Unknown"
                    
                self.place_names_tree.insert('', 'end', values=(
                    char.get('name', ''),
                    confidence,
                    location
                ))
                
            # Display survey numbers
            for num in numbers:
                confidence = num.get('confidence', 0)
                if isinstance(confidence, (int, float)):
                    confidence = f"{float(confidence):.2%}"
                location = num.get('location', (0, 0))
                if isinstance(location, (list, tuple)) and len(location) == 2:
                    location = f"({location[0]}, {location[1]})"
                else:
                    location = "Unknown"
                    
                self.survey_tree.insert('', 'end', values=(
                    num.get('number', ''),
                    confidence,
                    location
                ))
                
            # Display map features
            for category, items in symbols.items():
                if not isinstance(items, (list, tuple)):
                    continue
                    
                for item in items:
                    if not isinstance(item, dict):
                        continue
                        
                    if category == 'water':
                        area = item.get('area', 'N/A')
                        if isinstance(area, (int, float)):
                            area = f"{float(area):.2f} pixels"
                        self.features_tree.insert('', 'end', values=(
                            'Water',
                            'Water Body',
                            f"Area: {area}"
                        ))
                    elif category == 'terrain':
                        area = item.get('area', 'N/A')
                        if isinstance(area, (int, float)):
                            area = f"{float(area):.2f} pixels"
                        self.features_tree.insert('', 'end', values=(
                            'Terrain',
                            'Terrain Feature',
                            f"Area: {area}"
                        ))
                    elif category == 'transport':
                        length = item.get('length', 'N/A')
                        if isinstance(length, (int, float)):
                            length = f"{float(length):.2f} pixels"
                        self.features_tree.insert('', 'end', values=(
                            'Transport',
                            'Road/Railway',
                            f"Length: {length}"
                        ))
                            
            self.status_var.set(f"Processing completed - Found {len(characters)} place names, {len(numbers)} survey numbers")
            
        except Exception as e:
            self.show_error(f"Error displaying results: {str(e)}\n{traceback.format_exc()}")
        
    def show_error(self, message):
        """Display error message"""
        self.progress.stop()
        self.process_btn.state(['!disabled'])
        self.status_var.set("Error occurred")
        messagebox.showerror("Error", message)
        self.logger.error(message)

def main():
    root = tk.Tk()
    root.geometry("800x600")  # Set initial window size
    app = CadastralMapGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 