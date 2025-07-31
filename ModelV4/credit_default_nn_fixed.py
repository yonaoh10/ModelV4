import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from typing import List, Tuple, Optional
from tqdm import tqdm
import warnings

# התעלמות מאזהרות זמן ריצה של numpy (למשל, חלוקה באפס בחישוב מדדים)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. פונקציות עזר וטעינת נתונים ---

def load_and_prepare_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    טוענת נתונים מקובץ CSV באמצעות pandas ומפרידה למאפיינים (X) ותוויות (y).

    Args:
        filepath (str): הנתיב לקובץ ה-CSV.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: טאפל המכיל את מאפייני הנתונים (X) ואת תוויות המטרה (y).
        
    Raises:
        FileNotFoundError: אם הקובץ לא נמצא בנתיב שסופק.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"שגיאה: הקובץ לא נמצא בנתיב '{filepath}'")
        raise
        
    # המרה מפורשת לסוגי הנתונים הנכונים
    X = df.iloc[:, 1:-1].astype(np.float64)
    y = df.iloc[:, -1].astype(np.int64)
    return X, y

def balance_classes(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    מאזנת את המחלקות בסט הנתונים באמצעות דגימת יתר (oversampling) של מחלקת המיעוט.
    
    Args:
        X (np.ndarray): מאפייני הנתונים.
        y (np.ndarray): תוויות המטרה.
        random_state (int): זרע אקראי לשחזור תוצאות.

    Returns:
        Tuple[np.ndarray, np.ndarray]: נתונים ותוויות מאוזנים ומעורבבים.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) != 2:
        print("אזהרה: פונקציית האיזון מיועדת לסיווג בינארי אך זוהו מספר מחלקות שונה מ-2.")
        return X, y

    minority_class_label = unique_classes[np.argmin(counts)]
    majority_class_label = unique_classes[np.argmax(counts)]

    X_minority = X[y == minority_class_label]
    X_majority = X[y == majority_class_label]
    
    n_majority_samples = len(X_majority)

    # דגימת יתר של מחלקת המיעוט כדי להשתוות בגודלה למחלקת הרוב
    X_minority_resampled = resample(
        X_minority, 
        replace=True, 
        n_samples=n_majority_samples, 
        random_state=random_state
    )

    X_balanced = np.vstack((X_majority, X_minority_resampled))
    y_balanced = np.hstack((
        np.full(n_majority_samples, majority_class_label),
        np.full(n_majority_samples, minority_class_label)
    ))
    
    # ערבוב הנתונים המאוזנים למניעת הטיה בסדר הדגימות
    indices = np.arange(len(X_balanced))
    np.random.shuffle(indices)
    return X_balanced[indices], y_balanced[indices]

# --- 2. מחלקת הרשת העצבית ---

class NeuralNet:
    """
    מחלקה המממשת רשת עצבית פשוטה לסיווג בינארי עם יכולות אימון,
    חיזוי, רגולריזציה (L2), Dropout, ועצירה מוקדמת.
    """
    def __init__(self, sizes: List[int], lr: float = 0.01, reg: float = 0.001, dropout_rate: float = 0.2):
        """
        אתחול הרשת העצבית.
        Args:
            sizes (List[int]): רשימה של מספר הנוירונים בכל שכבה (כולל שכבת הקלט והפלט).
            lr (float): קצב הלמידה (learning rate).
            reg (float): מקדם הרגולריזציה (L2).
            dropout_rate (float): אחוז הנוירונים להשמטה בכל שכבת hidden במהלך האימון.
        """
        self.sizes = sizes
        self.lr = lr
        self.reg = reg
        self.dropout_rate = dropout_rate
        self.keep_prob = 1 - self.dropout_rate

        # אתחול משקולות בשיטת He, המתאימה לפונקציית אקטיבציה ReLU
        self.weights = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2 / sizes[i]) for i in range(len(sizes)-1)]
        self.biases = [np.zeros(sizes[i+1]) for i in range(len(sizes)-1)]

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """פונקציית אקטיבציה ReLU."""
        return np.maximum(0, x)

    def _drelu(self, x: np.ndarray) -> np.ndarray:
        """נגזרת של פונקציית ReLU."""
        return (x > 0).astype(float)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """פונקציית אקטיבציה סיגמואיד, עם הגנה מפני הצפה (overflow)."""
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        מבצע מעבר קדימה (forward pass) ברשת.
        
        Args:
            x (np.ndarray): נתוני הקלט.
            training (bool): דגל המציין אם הרשת במצב אימון (להפעלת Dropout).
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: טאפל של רשימות:
            - רשימת האקטיבציות בכל שכבה (כולל הקלט).
            - רשימת ערכי ה-z (הפלט הלינארי) לפני פונקציית האקטיבציה.
        """
        activations = [x]
        z_values = []
        
        # שכבות נסתרות (Hidden Layers) עם ReLU ו-Dropout
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            activation = self._relu(z)
            
            # החלת Inverted Dropout רק במצב אימון
            if training and self.dropout_rate > 0:
                # יצירת מסכה להשמטת נוירונים
                mask = (np.random.rand(*activation.shape) > self.dropout_rate).astype(float)
                # החלת המסכה וביצוע scaling כדי לפצות על הנוירונים שהושמטו
                activation = (activation * mask) / self.keep_prob
            
            activations.append(activation)
            
        # שכבת פלט (Output Layer) עם Sigmoid
        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_values.append(z_out)
        activations.append(self._sigmoid(z_out))
        
        return activations, z_values

    def backward(self, activations: List[np.ndarray], z_values: List[np.ndarray], y_true: np.ndarray):
        """מבצע מעבר אחורה (backpropagation) ומעדכן את המשקולות וההטיות."""
        batch_size = len(y_true)
        y_true = y_true.reshape(-1, 1)

        # חישוב השגיאה בשכבת הפלט
        delta = (activations[-1] - y_true)
        
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # גרדיאנטים עבור שכבת הפלט
        grads_w[-1] = (activations[-2].T @ delta / batch_size) + self.reg * self.weights[-1]
        grads_b[-1] = np.sum(delta, axis=0) / batch_size
        
        # הפצת השגיאה אחורה לשכבות הנסתרות
        for l in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[l+1].T) * self._drelu(z_values[l])
            grads_w[l] = (activations[l].T @ delta / batch_size) + self.reg * self.weights[l]
            grads_b[l] = np.sum(delta, axis=0) / batch_size
            
        # עדכון המשקולות וההטיות
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    def binary_cross_entropy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        חישוב פונקציית האובדן (Binary Cross-Entropy) כולל רכיב הרגולריזציה.
        """
        epsilon = 1e-9 # ערך קטן למניעת log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # אובדן Cross-Entropy
        data_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # אובדן רגולריזציה L2
        l2_reg_loss = 0.5 * self.reg * sum(np.sum(w*w) for w in self.weights)
        
        return float(data_loss + l2_reg_loss)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int, batch_size: int, patience: int):
        """
        מאמן את הרשת תוך שימוש בעצירה מוקדמת (Early Stopping).
        """
        best_val_loss = np.inf
        patience_counter = 0
        best_weights: Optional[List[np.ndarray]] = None
        best_biases: Optional[List[np.ndarray]] = None

        # שימוש ב-tqdm להצגת סרגל התקדמות
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X_train[indices], y_train[indices]
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                
                activations, z_values = self.forward(X_batch, training=True)
                self.backward(activations, z_values, y_batch)
            
            # הערכת ביצועים על סט הוולידציה
            val_preds = self.predict(X_val)
            val_loss = self.binary_cross_entropy(val_preds, y_val)
            
            # בדיקת מנגנון עצירה מוקדמת
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                    if best_weights and best_biases:
                        self.weights, self.biases = best_weights, best_biases
                    return

            # הדפסת סטטוס תקופתית
            if (epoch + 1) % 10 == 0:
                train_preds = self.predict(X_train)
                train_loss = self.binary_cross_entropy(train_preds, y_train)
                val_acc = np.mean((val_preds > 0.5) == y_val)
                tqdm.write(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}")
        
        # שמירת המשקולות הטובות ביותר בסוף האימון
        if best_weights and best_biases:
            self.weights, self.biases = best_weights, best_biases

    def predict(self, X: np.ndarray) -> np.ndarray:
        """מבצע חיזוי על נתונים חדשים."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # קריאה ל-forward במצב 'חיזוי' (training=False) כדי לבטל Dropout
        return self.forward(X, training=False)[0][-1].flatten()


# --- 3. פונקציית הערכה ---

def evaluate(model: NeuralNet, X: np.ndarray, y: np.ndarray):
    """מחשבת ומדפיסה מדדי ביצועים שונים עבור המודל."""
    preds = model.predict(X)
    preds_class = (preds > 0.5).astype(int)
    
    acc = np.mean(preds_class == y)
    tp = np.sum((preds_class == 1) & (y == 1))
    fp = np.sum((preds_class == 1) & (y == 0))
    fn = np.sum((preds_class == 0) & (y == 1))
    tn = np.sum((preds_class == 0) & (y == 0))
    
    # חישוב מדדים עם הגנה מפני חלוקה באפס
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50 + "\nFinal Evaluation on Test Set:\n" + "="*50)
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall (Sensitivity): {recall:.2%}")
    print(f"F1-Score: {f1:.4f}")
    print("="*50)
    
    print("\nConfusion Matrix:")
    print(f"{'':<15}{'Predicted 0':<15}{'Predicted 1':<15}")
    print(f"{'Actual 0':<15}{tn:<15}{fp:<15}")
    print(f"{'Actual 1':<15}{fn:<15}{tp:<15}")
    print("="*50)

# --- 4. זרימה עיקרית ---

if __name__ == "__main__":
    # הגדרות והיפר-פרמטרים
    FILENAME = 'Synthetic_Credit_Default_Data.csv'
    TEST_SIZE = 0.15
    VALIDATION_SIZE = 0.15
    RANDOM_STATE = 42
    
    # 1. טעינת וניתוח הנתונים
    X, y = load_and_prepare_data(FILENAME)
    print("="*50 + "\nData Analysis (Before Balancing):\n" + "="*50)
    print(f"Total samples: {len(y)}")
    print(f"Default (1) samples: {y.sum()} ({y.mean():.2%})")
    print(f"No Default (0) samples: {len(y) - y.sum()} ({1-y.mean():.2%})\n")

    # 2. פיצול הנתונים (70% אימון, 15% ולידציה, 15% מבחן)
    # שימוש ב-stratify כדי לשמור על יחס המחלקות המקורי בכל הסטים
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VALIDATION_SIZE), random_state=RANDOM_STATE, stratify=y
    )
    # פיצול הסט הזמני לוולידציה ומבחן
    val_test_ratio = TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Train set: {len(y_train)} samples")
    print(f"Validation set: {len(y_val)} samples")
    print(f"Test set: {len(y_test)} samples\n")
    
    # 3. איזון מחלקות (מבוצע *רק* על סט האימון למניעת דליפת מידע)
    print("Balancing training set classes...")
    X_train_bal, y_train_bal = balance_classes(X_train.values, y_train.values, random_state=RANDOM_STATE)
    print(f"Balanced training samples: {len(y_train_bal)}")
    print(f"Default (1) after balancing: {np.sum(y_train_bal)} ({np.mean(y_train_bal):.2%})\n")
    
    # 4. נרמול נתונים (Scaling)
    # יצירת Scaler והתאמתו (fit) *אך ורק* לנתוני האימון
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train_bal)
    # שימוש ב-Scaler המאומן כדי לבצע transform על סטי הוולידציה והמבחן
    X_val_norm = scaler.transform(X_val.values)
    X_test_norm = scaler.transform(X_test.values)
    print("Data scaled using MinMaxScaler (fitted on training data only).")

    # 5. בנייה ואימון המודל
    input_size = X_train_norm.shape[1]
    nn_sizes = [input_size, 32, 16, 1]
    model = NeuralNet(sizes=nn_sizes, lr=0.01, reg=0.001, dropout_rate=0.2)
    
    model.train(
        X_train_norm, y_train_bal, 
        X_val_norm, y_val.values,
        epochs=500, batch_size=64, patience=100
    )
    
    # 6. הערכת המודל על סט המבחן
    evaluate(model, X_test_norm, y_test.values)
    
    # 7. הדגמת חיזויים על דגימות אקראיות מסט המבחן
    print("\n" + "="*50 + "\nPrediction Examples from Test Set:\n" + "="*50)
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    for idx, i in enumerate(sample_indices):
        pred_value = model.predict(X_test_norm[i])[0]
        actual = y_test.iloc[i]
        pred_class = "ברירת מחדל" if pred_value > 0.5 else "שולם"
        actual_class = "ברירת מחדל" if actual == 1 else "שולם"
        mark = '✓' if pred_class == actual_class else '✗'
        print(f"דוגמה {idx+1}: חזוי: {pred_value:.4f} ({pred_class}), ממשי: {actual_class} {mark}")

