import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from DT_ui import Ui_DTWindow

from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QVBoxLayout, QPushButton,  QLabel,  QLineEdit, QComboBox, QFileDialog
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from sklearn.metrics import auc, RocCurveDisplay, roc_curve
import joblib



class DTWindow(QMainWindow, Ui_DTWindow):
    def __init__(self, dataset):
        super(DTWindow, self).__init__()
        self.setupUi(self)
        self.dataset = dataset

        self.setSizePushButton.clicked.connect(self.size_conjuntos) 
        self.EntrenarPushButton.clicked.connect(self.entrenar_modelo)   
        self.MConfusionPushButton.clicked.connect(self.mostrar_graficas)   
        self.descargaPushButton.clicked.connect(self.descargar_modelo)  

        self.model_dt = None
        self.conf_matrix = None

    def load_data(self):
        data = self.dataset
        self.feature_names = data.columns.tolist()[3:-1]

        # Convertir a variables categóricas a numéricas las columnas 'Sexo' y 'Emocion'
        data['sexo'] = data['sexo'].map({'hombre': 1, 'mujer': 0})
        data['emocion'] = data['emocion'].map({'Neutral': 0, 'Enfadado': 1, 'Triste': 2, 'Feliz': 3})

        x = data.iloc[:, 3:-1].values
        y = data.iloc[:, -1].values
        return x, y
    
    # Función para obtener el tamaño de los conjuntos de entrenamiento y prueba
    def size_conjuntos(self):
        size_value = int(self.sizeConjuntoLineEdit.text())  
        if 1 < size_value <= 100:
            x, y = self.load_data()
            train_size = size_value / 100
            self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(x, y, train_size)

            self.trainValueLabel.setText(str(len(self.x_train)))
            self.testValueLabel.setText(str(len(self.x_test)))
        else:
            QMessageBox.warning(self, "Valor inválido", "Introduce un valor entre 1 y 100.")

    # Función dividir el conjunto de entrenamiento (mediante slipt de scikit-learn)
    def split_data(self, x, y, train_size):
        
        test_size = 1 - train_size 
        # Aplicamos la funcion train_test_split() de scikit-learn para divir el conjunto de datos en train y test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=30) 
 
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        # Normalizamos los datos
        sc = StandardScaler()
        sc.fit(self.x_train)
        self.x_train_norm = sc.transform(self.x_train)
        self.x_test_norm = sc.transform(self.x_test)

        return self.x_train_norm, self.x_test_norm, self.y_train, self.y_test
    
    
    #Definimos el modelo de árbol de decisión
    def build_model(self):

        
        criterion = self.criterioComboBox.currentText() #Criterio del arbol (gini, entropy o log_loss)
        splitter = self.splitterComboBox.currentText()  #escoger el split (best o random)
        profundidad_max = self.maxDepthLineEdit.text()  #profundidad máxima del árbol
        
        if profundidad_max == "":
            max_depth = None
        else:
            max_depth = int(profundidad_max)
 
        self.model_dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
        

    #Funcion para entrenar el modelo (conectada al botón entrenar)
    def entrenar_modelo(self):
        # Asegurarse de que el modelo se construya antes de entrenar
        self.build_model()

        if self.model_dt:
            self.model_dt.fit(self.x_train_norm, self.y_train) 
            #visualizar el arbol de decisión creado
            plt.figure(figsize=(15, 10)) 
            tree.plot_tree(self.model_dt, class_names=['Neutral', 'Enfadado', 'Triste', 'Feliz']) 
            plt.show()
        else:
            QMessageBox.warning(self, "Error", "Primero debes construir el modelo.")


        self.mostrar_resultados()

    def mostrar_resultados(self):

        # Predicciones del modelo
        y_pred = self.model_dt.predict(self.x_test_norm)

        #Obtenemos la matriz de confusión del modelo (convertimos los resultados a text para mostrarlo)
        self.conf_matrix = confusion_matrix(self.y_test, y_pred) #labels=['Neutral', 'Enfadado', 'Triste', 'Feliz']
        conf_matrix_text = f"\nMatriz de Confusión:\n{self.conf_matrix}\n"
        self.processEntrenoTextEdit.append(conf_matrix_text)

        # Precision del modelo con los datos de Test
        scoresTest = self.model_dt.score(self.x_test_norm, self.y_test)
        resultadoTest = f"Test Accuracy: {scoresTest*100:.4f}%\nTest Error: {100-scoresTest*100:.4f}%"
        self.processEntrenoTextEdit.append("\nEvaluación del modelo con los datos de Test:\n" + resultadoTest)

        #Informe de clasificación
        class_names = ['Neutral', 'Enfadado', 'Triste', 'Feliz']
        class_report = classification_report(self.y_test, y_pred, target_names=class_names)
        class_report_text = f"\nInforme de Clasificación:\n{class_report}\n"
        self.processEntrenoTextEdit.append(class_report_text)

        #precisión general del modelo
        accuracy = accuracy_score(self.y_test, y_pred)
        accuracy_text = f"General Accuracy score: {accuracy*100:.2f}%"
        self.processEntrenoTextEdit.append(accuracy_text)
        
    # Funcion para mostrar la matriz de confusion y las curvas ROC
    def mostrar_graficas(self):
        plt.figure(figsize=(11, 7))
        sns.heatmap(self.conf_matrix, annot=True, fmt='4d')
        plt.show()

        y_test_onehot = pd.get_dummies(self.y_test)
        #Obtenemos las probabilidades de las clases para las curvas ROC
        y_pred_proba_clf = self.model_dt.predict_proba(self.x_test_norm)

        # CURVA ROC para Árbol de decisión
        fig, ax = plt.subplots(figsize=(10, 6))

        class_names = ['Neutral', 'Enfadado', 'Triste', 'Feliz']
        for i, class_name in enumerate(class_names):
            fpr, tpr, thresholds = roc_curve(y_test_onehot.iloc[:, i], y_pred_proba_clf[:, i])
            roc_auc = auc(fpr, tpr)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=class_name).plot(ax=ax)

        ax.set_title('Curva ROC')
        plt.show()

        # Gráfica de la importancia de cada caraterísticas mediante permutación
        
        result = permutation_importance(self.model_dt, self.x_test_norm, self.y_test, n_repeats=10, random_state=42, n_jobs=2)

        clf_importances = pd.Series(result.importances_mean, index=self.feature_names)
        
        fig, ax = plt.subplots()
        clf_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Importancia de las características utilizando permutación en el modelo Árbol de Decisión")
        ax.set_ylabel("Disminución de la precisión media")
        ax.set_xlabel("Características")
        fig.tight_layout()
        plt.show()



    def descargar_modelo(self): #Guardar el modelo entrenado
        if self.model_dt:
            file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo", "", "PKL Files (*.pkl);;All Files (*)")
            if file_path:
                joblib.dump(self.model_dt, file_path)
                QMessageBox.information(self, "Modelo Guardado", f"El modelo ha sido guardado en {file_path}.")
        else:
            QMessageBox.warning(self, "Error", "Primero debes entrenar el modelo.")
