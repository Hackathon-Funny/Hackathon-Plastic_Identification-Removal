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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
# MICROPLASTIC DETECTION CLASS (GEOMETRIC CLASSIFICATION ONLY)
# =============================================================================
# =============================================================================
# FAST ML CLASSIFIER (NO TENSORFLOW - TRAINS IN SECONDS!)
# =============================================================================

class FastMicroplasticClassifier:
    """Fast ML using Random Forest - trains in seconds on Pi 3!"""
    
    def __init__(self, model_path="microplastic_rf_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.classes = ['PE', 'PP', 'PET', 'PVC', 'PS', 'Fiber', 'Fragment', 'Unknown']
        self.confidence_threshold = 0.5
    
    def extract_features(self, particle_image):
        """Extract handcrafted features - FAST!"""
        if particle_image.size == 0 or particle_image.shape[0] < 5 or particle_image.shape[1] < 5:
            return None
        
        img = cv2.resize(particle_image, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        features = []
        
        # Shape features (10 features)
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments[:7])
        else:
            features.extend([0] * 7)
        
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / (h + 1e-5)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-5)
            features.extend([circularity, aspect_ratio, solidity])
        else:
            features.extend([0, 0, 0])
        
        # Color features (12 features)
        if len(img.shape) == 3:
            for channel in range(3):
                features.append(np.mean(img[:, :, channel]))
                features.append(np.std(img[:, :, channel]))
        else:
            features.extend([np.mean(gray), np.std(gray)] * 3)
        
        # HSV features (6 features)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if len(img.shape) == 3 else cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        for channel in range(3):
            features.append(np.mean(hsv[:, :, channel]))
            features.append(np.std(hsv[:, :, channel]))
        
        # Texture features (16 features)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-5)
        features.extend(hist)
        
        # Edge features (4 features)
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.mean(edges))
        features.append(np.std(edges))
        features.append(np.sum(edges > 0) / (edges.size + 1e-5))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features.append(np.mean(np.sqrt(sobelx**2 + sobely**2)))
        
        return np.array(features)
    
    def prepare_training_data(self, data_dir):
        """Load images and extract features"""
        X, y = [], []
        print("\n=== EXTRACTING FEATURES ===")
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"{class_name}: {len(images)} images")
            
            for img_file in images:
                img = cv2.imread(os.path.join(class_path, img_file))
                if img is None:
                    continue
                features = self.extract_features(img)
                if features is not None:
                    X.append(features)
                    y.append(class_idx)
        
        return np.array(X), np.array(y)
    
    def train(self, data_dir, n_estimators=100):
        """Train Random Forest - FAST (5-30 seconds on Pi 3!)"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        X, y = self.prepare_training_data(data_dir)
        if len(X) == 0:
            print("✗ No training data!")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"\nTraining: {len(X_train)} samples, Testing: {len(X_test)} samples")
        print(f"Training Random Forest ({n_estimators} trees)...")
        
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=20, 
                                           min_samples_split=5, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        print(f"\n✓ Training complete!")
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")
        
        y_pred = self.model.predict(X_test)
        existing_classes = [self.classes[i] for i in np.unique(y_test)]
        print("\n" + classification_report(y_test, y_pred, target_names=existing_classes, labels=np.unique(y_test)))
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {self.model_path}")
        
        return train_acc, test_acc
    
    def load_model(self):
        """Load trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from {self.model_path}")
            return True
        except:
            return False
    
    def predict_particle(self, particle_image):
        """Classify particle - FAST (<10ms on Pi 3!)"""
        if self.model is None:
            if not self.load_model():
                return None
        
        features = self.extract_features(particle_image)
        if features is None:
            return None
        
        features = features.reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        confidence = probabilities[prediction]
        predicted_class = self.classes[prediction] if confidence >= self.confidence_threshold else "Unknown"
        
        return {'class': predicted_class, 'confidence': float(confidence)}

def prepare_training_dataset():
    """Help organize training data"""
    print("\n=== DATASET PREPARATION ===")
    print("Organize images like this:")
    print("  dataset/train/")
    print("    PE/img1.jpg")
    print("    PP/img1.jpg")
    print("    PET/...")
    print("\nNeed 30-50 images per class minimum")

# =============================================================================
# MODIFIED MICROPLASTIC DETECTOR CLASS WITH CNN INTEGRATION
# REPLACE your existing MicroplasticDetector class with this version
# =============================================================================

class MicroplasticDetector:
    def __init__(self, db_path="experiment_data.db", use_cnn=True):
        self.db_path = db_path
        self.setup_database()
        self.MIN_AREA = 10
        self.MAX_AREA = 10000
        self.THRESHOLD = 127
        self.use_binary_split = False
        
        # CNN INTEGRATION - NEW!
        self.use_cnn = use_cnn
        self.cnn_classifier = None
        if use_cnn:
            self.cnn_classifier = FastMicroplasticClassifier()
            if os.path.exists("microplastic_rf_model.pkl"):
                self.cnn_classifier.load_model()
                print("✓ CNN classifier loaded")
            else:
                print("⚠ No trained CNN model found. Using geometric classification.")
                print("  Train a model using menu option 8 (Train ML Model)")
                self.use_cnn = False
        
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # MODIFIED TABLE - Added 'classification_method' and individual plastic types
        c.execute('''CREATE TABLE IF NOT EXISTS measurements
                     (id INTEGER PRIMARY KEY,
                      timestamp TEXT,
                      experiment_id TEXT,
                      total_particles INTEGER,
                      fibers INTEGER,
                      fragments INTEGER,
                      microplastics INTEGER,
                      pe_count INTEGER,
                      pp_count INTEGER,
                      pet_count INTEGER,
                      pvc_count INTEGER,
                      ps_count INTEGER,
                      unknown_count INTEGER,
                      avg_particle_size REAL,
                      coagulant_type TEXT,
                      coagulant_dose REAL,
                      ph REAL,
                      mixing_speed INTEGER,
                      settling_time INTEGER,
                      stage TEXT,
                      classification_method TEXT)''')
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
        
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 5)
        _, bright = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(adaptive, bright)
        
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
    
    def extract_particle_image(self, frame, contour, padding=5):
        """Extract individual particle image for CNN classification"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        particle_img = frame[y1:y2, x1:x2]
        return particle_img
    
    def detect_and_count(self, frame):
        """MODIFIED: Now uses CNN for classification if available"""
        
        # Step 1: Detection (find particles) - unchanged
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
        
        # Initialize results with CNN categories
        results = {
            'particles': 0,
            'fibers': 0,
            'fragments': 0,
            'pe': 0,      # NEW
            'pp': 0,      # NEW
            'pet': 0,     # NEW
            'pvc': 0,     # NEW
            'ps': 0,      # NEW
            'unknown': 0, # NEW
            'total_area': 0,
            'detections': [],
            'binary_methods': binary_methods,
            'classification_method': 'CNN' if self.use_cnn and self.cnn_classifier else 'Geometric'
        }
        
        # Step 2: Classification (identify particle type)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # CNN CLASSIFICATION (NEW!)
            if self.use_cnn and self.cnn_classifier and self.cnn_classifier.model is not None:
                particle_img = self.extract_particle_image(frame, contour)
                
                if particle_img.size > 0:  # Make sure we extracted something
                    prediction = self.cnn_classifier.predict_particle(particle_img)
                    
                    if prediction:
                        label = prediction['class']
                        confidence = prediction['confidence']
                        
                        # Count by plastic type
                        if label == 'PE':
                            results['pe'] += 1
                            color = (255, 0, 0)  # Red
                        elif label == 'PP':
                            results['pp'] += 1
                            color = (0, 255, 0)  # Green
                        elif label == 'PET':
                            results['pet'] += 1
                            color = (0, 0, 255)  # Blue
                        elif label == 'PVC':
                            results['pvc'] += 1
                            color = (255, 255, 0)  # Cyan
                        elif label == 'PS':
                            results['ps'] += 1
                            color = (255, 0, 255)  # Magenta
                        elif label == 'Fiber':
                            results['fibers'] += 1
                            color = (255, 165, 0)  # Orange
                        elif label == 'Fragment':
                            results['fragments'] += 1
                            color = (0, 165, 255)  # Orange-red
                        else:  # Unknown
                            results['unknown'] += 1
                            color = (128, 128, 128)  # Gray
                        
                        # Add confidence to label
                        display_label = f"{label} ({confidence:.2f})"
                    else:
                        # CNN failed, fallback to geometric
                        label, color = self.geometric_classification(aspect_ratio, area)
                        display_label = label
                        self.update_geometric_counts(label, results)
                else:
                    # Could not extract image, use geometric
                    label, color = self.geometric_classification(aspect_ratio, area)
                    display_label = label
                    self.update_geometric_counts(label, results)
            
            else:
                # GEOMETRIC CLASSIFICATION (FALLBACK)
                label, color = self.geometric_classification(aspect_ratio, area)
                display_label = label
                self.update_geometric_counts(label, results)
            
            results['total_area'] += area
            results['detections'].append({
                'label': display_label,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'color': color
            })
        
        # Update old category counts for compatibility
        results['particles'] = results['pe'] + results['pp'] + results['pet'] + results['pvc'] + results['ps'] + results['unknown']
        
        return results, thresh
    
    def geometric_classification(self, aspect_ratio, area):
        """Original geometric classification as fallback"""
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            return "fiber", (255, 165, 0)  # Orange
        elif area < 200:
            return "microplastic", (255, 0, 0)  # Red
        else:
            return "fragment", (0, 0, 255)  # Blue
    
    def update_geometric_counts(self, label, results):
        """Update counts for geometric classification"""
        if label == "fiber":
            results['fibers'] += 1
        elif label == "microplastic":
            results['unknown'] += 1  # Map to unknown since we don't know type
        elif label == "fragment":
            results['fragments'] += 1
    
    def save_measurement(self, results, experiment_params):
        """MODIFIED: Save CNN classification results"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        total = (results['pe'] + results['pp'] + results['pet'] + 
                results['pvc'] + results['ps'] + results['fibers'] + 
                results['fragments'] + results['unknown'])
        avg_size = results['total_area'] / total if total > 0 else 0
        
        c.execute('''INSERT INTO measurements 
                     (timestamp, experiment_id, total_particles, fibers, 
                      fragments, microplastics, pe_count, pp_count, pet_count,
                      pvc_count, ps_count, unknown_count, avg_particle_size,
                      coagulant_type, coagulant_dose, ph, mixing_speed,
                      settling_time, stage, classification_method)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(),
                   experiment_params.get('exp_id', 'unknown'),
                   total,
                   results['fibers'],
                   results['fragments'],
                   results['particles'],
                   results.get('pe', 0),
                   results.get('pp', 0),
                   results.get('pet', 0),
                   results.get('pvc', 0),
                   results.get('ps', 0),
                   results.get('unknown', 0),
                   avg_size,
                   experiment_params.get('coagulant_type'),
                   experiment_params.get('coagulant_dose'),
                   experiment_params.get('ph'),
                   experiment_params.get('mixing_speed'),
                   experiment_params.get('settling_time'),
                   experiment_params.get('stage', 'before'),
                   results.get('classification_method', 'Geometric')))
        conn.commit()
        conn.close()
        
        print(f"✓ Measurement saved! Total particles: {total}")
        if results.get('classification_method') == 'CNN':
            print(f"  PE: {results['pe']}, PP: {results['pp']}, PET: {results['pet']}")
            print(f"  PVC: {results['pvc']}, PS: {results['ps']}, Unknown: {results['unknown']}")


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
    cv2.resizeWindow("1. Detection Results", 900, 700)
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
# ============================================================================

def main_menu():
    detector = MicroplasticDetector(use_cnn=True)
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
        
        print("\n🚀 FAST ML (NO TENSORFLOW):")
        print("  8. Prepare Training Dataset")
        print("  9. Train ML Model (Fast!)")
        print("  10. Toggle ML On/Off")
        
        print("\nOTHER:")
        print("  11. Exit")
        
        choice = input("\nSelect option (1-11): ").strip()
        
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
            prepare_training_dataset()
        
        elif choice == '9':
            print("\n=== TRAIN ML MODEL ===")
            train_dir = input("Enter path to training data folder: ").strip()
            if not os.path.exists(train_dir):
                print("✗ Directory not found")
                continue
            
            trees = int(input("Number of trees (recommended 100): ") or "100")
            
            print("\n🚀 Starting training (this is FAST - takes seconds!)...")
            ml_classifier = FastMicroplasticClassifier()
            ml_classifier.train(train_dir, n_estimators=trees)
            
            # Reload detector with new model
            detector.cnn_classifier = ml_classifier
            detector.use_cnn = True
            print("✓ Detector updated with new model")
        
        elif choice == '10':
            detector.use_cnn = not detector.use_cnn
            status = "ENABLED" if detector.use_cnn else "DISABLED"
            print(f"\n✓ ML classification {status}")
            if detector.use_cnn and not os.path.exists("microplastic_rf_model.pkl"):
                print("⚠ Warning: No trained model found. Train one first (option 9)")
                detector.use_cnn = False
        
        elif choice == '11':
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid option. Please select 1-11.")
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


