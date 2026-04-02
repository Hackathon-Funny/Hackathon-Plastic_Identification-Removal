import cv2
import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

# =============================================================================
# MICROPLASTIC DETECTION CLASS (GEOMETRIC CLASSIFICATION ONLY)
# =============================================================================
class MicroplasticDetector:
    def __init__(self, db_path="experiment_data.db"):
        self.db_path = db_path
        self.setup_database()
        self.MIN_AREA = 10
        self.MAX_AREA = 10000
        self.THRESHOLD = 127
        self.use_binary_split = False
        
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS measurements
                     (id INTEGER PRIMARY KEY,
                      timestamp TEXT,
                      experiment_id TEXT,
                      total_particles INTEGER,
                      fibers INTEGER,
                      fragments INTEGER,
                      microplastics INTEGER,
                      avg_particle_size REAL,
                      coagulant_type TEXT,
                      coagulant_dose REAL,
                      ph REAL,
                      mixing_speed INTEGER,
                      settling_time INTEGER,
                      stage TEXT)''')
        conn.commit()
        conn.close()
    
    def enable_binary_split(self, enabled=True):
        """Enable/disable binary pixel splitting feature"""
        self.use_binary_split = enabled
        print(f"Binary pixel splitting: {'ENABLED' if enabled else 'DISABLED'}")
    
    def binary_pixel_split(self, frame):
        """Split every pixel into pure black or white using multiple methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Method 1: Otsu's thresholding
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive threshold (THIS IS THE ONE WE USE FOR DETECTION)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 5)
        
        # Method 3: Detect bright particles
        _, bright = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Combine adaptive with bright detection
        combined = cv2.bitwise_or(adaptive, bright)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return adaptive, {
            'otsu': otsu,
            'adaptive': adaptive,
            'bright': bright,
            'combined': combined,
            'final': cleaned
        }
    
    def detect_and_count(self, frame):
        if self.use_binary_split:
            thresh, binary_methods = self.binary_pixel_split(frame)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 21, 5)
            
            _, thresh_bright = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_or(thresh, thresh_bright)
            
            binary_methods = None
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = {
            'particles': 0,
            'fibers': 0,
            'fragments': 0,
            'total_area': 0,
            'detections': [],
            'binary_methods': binary_methods
        }
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Geometric classification
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                label = "fiber"
                results['fibers'] += 1
                color = (255, 165, 0)  # Orange
            elif area < 200:
                label = "microplastic"
                results['particles'] += 1
                color = (255, 0, 0)  # Red
            else:
                label = "fragment"
                results['fragments'] += 1
                color = (0, 0, 255)  # Blue
            
            results['total_area'] += area
            results['detections'].append({
                'label': label,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'color': color
            })
        
        return results, thresh
    
    def save_measurement(self, results, experiment_params):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        total = results['particles'] + results['fibers'] + results['fragments']
        avg_size = results['total_area'] / total if total > 0 else 0
        
        c.execute('''INSERT INTO measurements 
                     (timestamp, experiment_id, total_particles, fibers, 
                      fragments, microplastics, avg_particle_size,
                      coagulant_type, coagulant_dose, ph, mixing_speed,
                      settling_time, stage)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(),
                   experiment_params.get('exp_id', 'unknown'),
                   total,
                   results['fibers'],
                   results['fragments'],
                   results['particles'],
                   avg_size,
                   experiment_params.get('coagulant_type'),
                   experiment_params.get('coagulant_dose'),
                   experiment_params.get('ph'),
                   experiment_params.get('mixing_speed'),
                   experiment_params.get('settling_time'),
                   experiment_params.get('stage', 'before')))
        conn.commit()
        conn.close()
        print(f"✓ Measurement saved! Total particles: {total}")

# =============================================================================
# DATA ANALYSIS CLASS
# =============================================================================
class DataManager:
    def __init__(self, db_path="experiment_data.db"):
        self.db_path = db_path
    
    def get_all_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM measurements", conn)
        conn.close()
        return df
    
    def export_for_analysis(self, filename="experiment_results.csv"):
        df = self.get_all_data()
        
        if len(df) == 0:
            print("ERROR: No data in database. Run option 1 or 8 first to collect data.")
            return None
            
        exp_ids = df['experiment_id'].unique()
        results = []
        
        for exp_id in exp_ids:
            exp_data = df[df['experiment_id'] == exp_id]
            before = exp_data[exp_data['stage'] == 'before']
            after = exp_data[exp_data['stage'] == 'after']
            
            if len(before) > 0 and len(after) > 0:
                before_count = before['total_particles'].values[0]
                after_count = after['total_particles'].values[0]
                efficiency = ((before_count - after_count) / before_count) * 100 if before_count > 0 else 0
                
                results.append({
                    'experiment_id': exp_id,
                    'coagulant_type': before['coagulant_type'].values[0],
                    'coagulant_dose': before['coagulant_dose'].values[0],
                    'ph': before['ph'].values[0],
                    'mixing_speed': before['mixing_speed'].values[0],
                    'settling_time': before['settling_time'].values[0],
                    'before_count': before_count,
                    'after_count': after_count,
                    'removal_efficiency': efficiency
                })
        
        if len(results) == 0:
            print("ERROR: No complete experiments (need both 'before' and 'after' stages).")
            return None
            
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)
        print(f"✓ Data exported to {filename}")
        print(f"  Total complete experiments: {len(results_df)}")
        return results_df
    
    def visualize_results(self, filename="experiment_results.csv"):
        try:
            df = pd.read_csv(filename)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("ERROR: No data to visualize. Export data first (it's empty).")
            return
            
        if len(df) == 0:
            print("ERROR: CSV file is empty. Collect data using option 1 or 8 first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].scatter(df['coagulant_dose'], df['removal_efficiency'], s=100, alpha=0.6)
        axes[0, 0].set_xlabel('Coagulant Dose (mg/L)')
        axes[0, 0].set_ylabel('Removal Efficiency (%)')
        axes[0, 0].set_title('Efficiency vs Coagulant Dose')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].scatter(df['ph'], df['removal_efficiency'], s=100, alpha=0.6)
        axes[0, 1].set_xlabel('pH')
        axes[0, 1].set_ylabel('Removal Efficiency (%)')
        axes[0, 1].set_title('Efficiency vs pH')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].scatter(df['mixing_speed'], df['removal_efficiency'], s=100, alpha=0.6)
        axes[1, 0].set_xlabel('Mixing Speed (rpm)')
        axes[1, 0].set_ylabel('Removal Efficiency (%)')
        axes[1, 0].set_title('Efficiency vs Mixing Speed')
        axes[1, 0].grid(True, alpha=0.3)
        
        df.groupby('coagulant_type')['removal_efficiency'].mean().plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_ylabel('Avg Removal Efficiency (%)')
        axes[1, 1].set_title('Coagulant Type Comparison')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('experimental_results.png', dpi=300)
        print("✓ Visualization saved as experimental_results.png")
        plt.show()

# =============================================================================
# PREDICTIVE MODEL CLASS (RandomForest ML)
# =============================================================================
class RemovalPredictor:
    def __init__(self):
        self.model = None
        self.features = ['coagulant_dose', 'ph', 'mixing_speed', 'settling_time']
    
    def train_model(self, data_file="experiment_results.csv"):
        try:
            df = pd.read_csv(data_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("ERROR: No data file. Run option 2 first.")
            return
            
        if len(df) == 0:
            print("ERROR: Data file is empty. Collect data first.")
            return
            
        X = df[self.features]
        y = df['removal_efficiency']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        print("\n=== MODEL PERFORMANCE ===")
        print(f"Training R²: {r2_score(y_train, train_pred):.3f}")
        print(f"Testing R²: {r2_score(y_test, test_pred):.3f}")
        print(f"Training MAE: {mean_absolute_error(y_train, train_pred):.2f}%")
        print(f"Testing MAE: {mean_absolute_error(y_test, test_pred):.2f}%")
        
        print("\n=== FEATURE IMPORTANCE ===")
        for feat, imp in zip(self.features, self.model.feature_importances_):
            print(f"{feat}: {imp:.3f}")
        
        with open('removal_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("\n✓ Model saved as removal_model.pkl")
    
    def predict(self, coagulant_dose, ph, mixing_speed, settling_time):
        if self.model is None:
            with open('removal_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        input_data = np.array([[coagulant_dose, ph, mixing_speed, settling_time]])
        return self.model.predict(input_data)[0]
    
    def optimize_conditions(self, data_file="experiment_results.csv"):
        try:
            df = pd.read_csv(data_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("ERROR: No data file. Run option 2 first.")
            return
            
        if len(df) == 0:
            print("ERROR: Data file is empty.")
            return
            
        best_idx = df['removal_efficiency'].idxmax()
        optimal = df.loc[best_idx]
        
        print("\n=== OPTIMAL CONDITIONS ===")
        print(f"Coagulant Dose: {optimal['coagulant_dose']:.1f} mg/L")
        print(f"pH: {optimal['ph']:.1f}")
        print(f"Mixing Speed: {optimal['mixing_speed']:.0f} rpm")
        print(f"Settling Time: {optimal['settling_time']:.0f} min")
        print(f"Removal Efficiency: {optimal['removal_efficiency']:.1f}%")
    
    def generate_response_surface(self, data_file="experiment_results.csv"):
        try:
            df = pd.read_csv(data_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("ERROR: No data file. Run option 2 first.")
            return
        
        if len(df) == 0:
            print("ERROR: Data file is empty.")
            return
        
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(df['coagulant_dose'], df['ph'], df['removal_efficiency'],
                   c=df['removal_efficiency'], cmap='viridis', s=100)
        ax1.set_xlabel('Dose (mg/L)')
        ax1.set_ylabel('pH')
        ax1.set_zlabel('Efficiency (%)')
        ax1.set_title('Response Surface: Dose & pH')
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(df['coagulant_dose'], df['mixing_speed'], df['removal_efficiency'],
                   c=df['removal_efficiency'], cmap='plasma', s=100)
        ax2.set_xlabel('Dose (mg/L)')
        ax2.set_ylabel('Mixing (rpm)')
        ax2.set_zlabel('Efficiency (%)')
        ax2.set_title('Response Surface: Dose & Mixing')
        
        plt.tight_layout()
        plt.savefig('response_surface.png', dpi=300)
        print("✓ Response surface saved")
        plt.show()

# =============================================================================
# IMAGE BATCH PROCESSING
# =============================================================================
def analyze_images_from_folder(detector):
    """Analyze already-taken microscope images"""
    
    print("\n=== ANALYZE IMAGES FROM FOLDER ===")
    
    use_binary = input("Enable advanced binary pixel splitting? (y/n): ").strip().lower()
    detector.enable_binary_split(use_binary == 'y')
    
    folder_path = input("Enter folder path with images (or press Enter for current folder): ").strip()
    
    if not folder_path:
        folder_path = "."
    
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder '{folder_path}' not found")
        return
    
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if len(image_files) == 0:
        print(f"ERROR: No image files found in '{folder_path}'")
        return
    
    image_files.sort()
    
    print(f"\nFound {len(image_files)} images")
    for i, img_file in enumerate(image_files):
        print(f"  {i+1}. {os.path.basename(img_file)}")
    
    print("\n=== SELECT IMAGES ===")
    print("Options:")
    print("  - Type image numbers separated by commas (e.g., 1,2,5,10)")
    print("  - Type range with dash (e.g., 1-5 for images 1 through 5)")
    print("  - Type 'all' to analyze all images")
    
    selection = input("\nYour selection: ").strip().lower()
    
    selected_indices = []
    
    if selection == 'all':
        selected_indices = list(range(len(image_files)))
    elif '-' in selection:
        try:
            start, end = selection.split('-')
            start = int(start.strip()) - 1
            end = int(end.strip())
            selected_indices = list(range(start, end))
        except:
            print("ERROR: Invalid range format")
            return
    else:
        try:
            numbers = [int(n.strip()) - 1 for n in selection.split(',')]
            selected_indices = numbers
        except:
            print("ERROR: Invalid selection format")
            return
    
    selected_indices = [i for i in selected_indices if 0 <= i < len(image_files)]
    
    if len(selected_indices) == 0:
        print("ERROR: No valid images selected")
        return
    
    selected_files = [image_files[i] for i in selected_indices]
    
    print(f"\nSelected {len(selected_files)} images:")
    for f in selected_files:
        print(f"  - {os.path.basename(f)}")
    
    print("\n=== EXPERIMENT SETUP ===")
    exp_id = input("Experiment ID (e.g., EXP001): ")
    coag_type = input("Coagulant type (chitosan/alum/banana): ")
    coag_dose = float(input("Coagulant dose (mg/L): "))
    ph = float(input("pH: "))
    mixing = int(input("Mixing speed (rpm): "))
    settling = int(input("Settling time (min): "))
    stage = input("Are these images BEFORE or AFTER treatment? (before/after): ").strip().lower()
    
    exp_params = {
        'exp_id': exp_id,
        'coagulant_type': coag_type,
        'coagulant_dose': coag_dose,
        'ph': ph,
        'mixing_speed': mixing,
        'settling_time': settling,
        'stage': stage
    }
    
    cv2.namedWindow("1. Detection Results", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. Adaptive Threshold (USED FOR DETECTION)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("3. Debug Window", cv2.WINDOW_NORMAL)
    
    cv2.moveWindow("1. Detection Results", 50, 50)
    cv2.moveWindow("2. Adaptive Threshold (USED FOR DETECTION)", 700, 50)
    cv2.moveWindow("3. Debug Window", 1350, 50)
    
    for img_path in selected_files:
        print(f"\n--- Processing: {os.path.basename(img_path)} ---")
        
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"ERROR: Could not read {img_path}")
            continue
        
        results, thresh = detector.detect_and_count(frame)
        
        display_frame = frame.copy()
        for detection in results['detections']:
            x, y, w, h = detection['bbox']
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), detection['color'], 2)
            cv2.putText(display_frame, detection['label'], (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection['color'], 2)
        
        total = results['particles'] + results['fibers'] + results['fragments']
        cv2.putText(display_frame, f"Total: {total}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("1. Detection Results", display_frame)
        cv2.waitKey(50)
        
        cv2.imshow("2. Adaptive Threshold (USED FOR DETECTION)", thresh)
        cv2.waitKey(50)
        
        if detector.use_binary_split and results.get('binary_methods'):
            debug_img = results['binary_methods'].get('combined', thresh)
        else:
            debug_img = thresh
        cv2.imshow("3. Debug Window", debug_img)
        cv2.waitKey(50)
        
        print(f"Particles detected: {total}")
        print(f"  Microplastics: {results['particles']}")
        print(f"  Fibers: {results['fibers']}")
        print(f"  Fragments: {results['fragments']}")
        
        save = input("Save this measurement? (y/n): ").strip().lower()
        if save == 'y':
            detector.save_measurement(results, exp_params)
        
        print("Press any key to continue to next image...")
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("\n✓ Finished processing selected images")

# =============================================================================
# MAIN MENU SYSTEM
# =============================================================================
def main_menu():
    detector = MicroplasticDetector()
    data_manager = DataManager()
    predictor = RemovalPredictor()
    
    while True:
        print("\n" + "="*60)
        print("MICROPLASTIC REMOVAL PROJECT")
        print("="*60)
        print("DETECTION:")
        print("  1. Run Detection (live camera)")
        print("  2. Analyze Images from Folder")
        print("\nDATA ANALYSIS:")
        print("  3. Analyze Data (export & visualize)")
        print("  4. Train Predictive Model (RandomForest)")
        print("  5. Find Optimal Conditions")
        print("  6. Generate Response Surface")
        print("  7. Make Prediction")
        print("\nOTHER:")
        print("  8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            run_detection(detector)
        elif choice == '2':
            analyze_images_from_folder(detector)
        elif choice == '3':
            result = data_manager.export_for_analysis()
            if result is not None:
                data_manager.visualize_results()
        elif choice == '4':
            predictor.train_model()
        elif choice == '5':
            predictor.optimize_conditions()
        elif choice == '6':
            predictor.generate_response_surface()
        elif choice == '7':
            try:
                dose = float(input("Coagulant dose (mg/L): "))
                ph = float(input("pH: "))
                speed = int(input("Mixing speed (rpm): "))
                time = int(input("Settling time (min): "))
                pred = predictor.predict(dose, ph, speed, time)
                print(f"\nPredicted removal efficiency: {pred:.1f}%")
            except:
                print("Error making prediction. Train model first (option 4)")
        elif choice == '8':
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid option. Please select 1-8.")

def run_detection(detector):
    """Live camera detection"""
    use_binary = input("\nEnable advanced binary pixel splitting for live detection? (y/n): ").strip().lower()
    detector.enable_binary_split(use_binary == 'y')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not found")
        return
    
    print("\n=== EXPERIMENT SETUP ===")
    exp_id = input("Experiment ID (e.g., EXP001): ")
    coag_type = input("Coagulant type (chitosan/alum/banana): ")
    coag_dose = float(input("Coagulant dose (mg/L): "))
    ph = float(input("pH: "))
    mixing = int(input("Mixing speed (rpm): "))
    settling = int(input("Settling time (min): "))
    stage = input("Stage (before/after): ")
    
    exp_params = {
        'exp_id': exp_id,
        'coagulant_type': coag_type,
        'coagulant_dose': coag_dose,
        'ph': ph,
        'mixing_speed': mixing,
        'settling_time': settling,
        'stage': stage
    }
    
    print("\nCamera opening...")
    print("Controls:")
    print("  Press 's' to SAVE measurement")
    print("  Press 'q' to QUIT")
    
    saved = False
    
    cv2.namedWindow("1. Detection Results", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. Adaptive Threshold (USED FOR DETECTION)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("3. Debug Window", cv2.WINDOW_NORMAL)
    
    cv2.moveWindow("1. Detection Results", 50, 50)
    cv2.moveWindow("2. Adaptive Threshold (USED FOR DETECTION)", 700, 50)
    cv2.moveWindow("3. Debug Window", 1350, 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Can't read from camera")
            break
        
        results, thresh = detector.detect_and_count(frame)
        
        display_frame = frame.copy()
        for detection in results['detections']:
            x, y, w, h = detection['bbox']
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), detection['color'], 2)
            cv2.putText(display_frame, detection['label'], (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection['color'], 2)
        
        total = results['particles'] + results['fibers'] + results['fragments']
        cv2.putText(display_frame, f"Total: {total}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if saved:
            cv2.putText(display_frame, "SAVED! Press Q to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("1. Detection Results", display_frame)
        cv2.imshow("2. Adaptive Threshold (USED FOR DETECTION)", thresh)
        
        if detector.use_binary_split and results.get('binary_methods'):
            debug_img = results['binary_methods'].get('combined', thresh)
        else:
            debug_img = thresh
        cv2.imshow("3. Debug Window", debug_img)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:
            print("Exiting camera...")
            break
        elif key == ord('s') and not saved:
            detector.save_measurement(results, exp_params)
            saved = True
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_menu()


