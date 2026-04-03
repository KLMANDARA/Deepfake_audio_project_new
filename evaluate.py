
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# dummy example (replace with real test loading)
y_true = np.random.randint(0,2,100)
y_pred = np.random.randint(0,2,100)

cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_true, y_pred))
