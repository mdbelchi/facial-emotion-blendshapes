import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QVBoxLayout, QPushButton,  QLabel,  QLineEdit, QComboBox, QFileDialog
from NN_ui import Ui_NNWindow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from sklearn.metrics import auc, RocCurveDisplay, roc_curve
from scikeras.wrappers import KerasClassifier
from sklearn.inspection import permutation_importance


class NNWindow(QMainWindow, Ui_NNWindow):
    def __init__(self, dataset):
        super(NNWindow, self).__init__()
        self.setupUi(self)
        self.dataset = dataset
        self.setSizePushButton.clicked.connect(self.size_conjuntos) 
        self.crearModeloPushButton.clicked.connect(self.parametros_modelo)  
        self.EntrenarPushButton.clicked.connect(self.entrenar_modelo)   
        self.MConfusionPushButton.clicked.connect(self.mostrar_graficas)   
        self.descargaPushButton.clicked.connect(self.descargar_modelo) 
        self.model = None
        self.conf_matrix = None

    def load_data(self):
        data = self.dataset
        self.feature_names = data.columns.tolist()[3:-1]

        data['sexo'] = data['sexo'].map({'hombre': 1, 'mujer': 0})
        data['emocion'] = data['emocion'].map({'Neutral': 0, 'Enfadado': 1, 'Triste': 2, 'Feliz': 3})

        x = data.iloc[:, 3:-1].values
        y = data.iloc[:, -1].values
        return x, y

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


    def split_data(self, x, y, train_size):
        test_size = 1 - train_size  # Calcular el tamaño del conjunto de test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=30) 
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        sc = StandardScaler()
        sc.fit(self.x_train)
        self.x_train_norm = sc.transform(self.x_train)
        self.x_test_norm = sc.transform(self.x_test)

        return self.x_train_norm, self.x_test_norm, self.y_train, self.y_test

    def parametros_modelo(self):
        num_capas = int(self.capasLineEdit.text())
        optimizer = self.optimizadorComboBox.currentText()
        epochs = int(self.epochsLineEdit.text())
        

        if num_capas > 0 and epochs > 0:
            self.capas_window = ConfiguraCapas(num_capas, parent=self)
            self.capas_window.show()
        else:
            QMessageBox.warning(self, "Valor inválido", "Introduce valores válidos para el número de capas y épocas.")
        self.optimizer = optimizer
        self.epochs = epochs


    def build_model(self, capas_config):

        tf.keras.backend.clear_session()
        n_cols = self.x_train_norm.shape[1] 
        n_clases = len(set(self.y_train))
        model = Sequential()    
        model.add(Input(shape=(n_cols,))) 

        for neuronas, activation in capas_config:
            model.add(Dense(units=neuronas, activation=activation))

        model.add(Dense(units=n_clases, activation='softmax'))
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

        self.model = model

    def entrenar_modelo(self):

        self.batch_size = int(self.batchSizeComboBox.currentText())
        if self.model:
            modelo = self.model.fit(self.x_train_norm, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.x_test_norm, self.y_test))
        else:
            QMessageBox.warning(self, "Error", "Primero debes construir el modelo.")
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

        ax1.plot(modelo.history['loss'], label='Training loss')
        ax1.plot(modelo.history['val_loss'], label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(modelo.history['sparse_categorical_accuracy'], label='Training accuracy')
        ax2.plot(modelo.history['val_sparse_categorical_accuracy'], label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        self.mostrar_resultados()

    def mostrar_resultados(self):
        model_summary = []
        self.model.summary(print_fn=lambda x: model_summary.append(x))
        summary_str = "\n".join(model_summary)
        self.processEntrenoTextEdit.append("Resumen del Modelo:\n" + summary_str)

        scores = self.model.evaluate(self.x_test_norm, self.y_test, verbose=0)
        resultado = f"Test Accuracy: {scores[1]*100:.4f}%\nTest Error: {100-scores[1]*100:.4f}%"
        self.processEntrenoTextEdit.append("\nEvaluación del modelo con los datos de Test:\n" + resultado)

        prediction = self.model.predict(self.x_test_norm)
        y_pred = prediction.argmax(axis=-1) 
        self.conf_matrix = confusion_matrix(self.y_test, y_pred)
        conf_matrix_text = f"\nMatriz de Confusión:\n{self.conf_matrix}\n"
        self.processEntrenoTextEdit.append(conf_matrix_text)

        target_names = ['Neutral', 'Enfadado', 'Triste', 'Feliz']
        class_report = classification_report(self.y_test, y_pred, target_names=target_names)
        class_report_text = f"\nInforme de Clasificación:\n{class_report}\n"
        self.processEntrenoTextEdit.append(class_report_text)

        accuracy = accuracy_score(self.y_test, y_pred)
        accuracy_text = f"General Accuracy score: {accuracy*100:.2f}%"
        self.processEntrenoTextEdit.append(accuracy_text)

    def mostrar_graficas(self):
        plt.figure(figsize=(11, 7))
        sns.heatmap(self.conf_matrix, annot=True, fmt='4d')
        plt.show()

        y_test_onehot = pd.get_dummies(self.y_test)
        y_scores_red_neuronal = self.model.predict(self.x_test_norm)

        fig, ax = plt.subplots(figsize=(10, 6))

        class_names = ['Neutral', 'Enfadado', 'Triste', 'Feliz']
        for i, class_name in enumerate(class_names):
            fpr, tpr, thresholds = roc_curve(y_test_onehot.iloc[:, i], y_scores_red_neuronal[:, i])
            roc_auc = auc(fpr, tpr)
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=class_name).plot(ax=ax)

        ax.set_title('Curva ROC')
        plt.show()

        ##Gráfica: permutation-features

        keras_clf = KerasClassifier(model=self.model, epochs=0, batch_size=32, verbose=0)
        keras_clf.fit(self.x_train_norm, self.y_train)
        perm_importance = permutation_importance(keras_clf, self.x_test_norm, self.y_test, n_repeats=10, random_state=42)
        #feature_names = [...]  
        sorted_idx = perm_importance.importances_mean.argsort()
        #sfeature_names = feature_names[3:-1]


        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(self.feature_names)[sorted_idx])
        plt.xlabel("Importancia de la característica")
        plt.title("Importancia de las características según Permutation Importance")
        plt.show()


    def descargar_modelo(self): #Guardar el modelo en formato .h5
        if self.model:
            file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo", "", "H5 Files (*.h5);;All Files (*)")
            if file_path:
                self.model.save(file_path)
                QMessageBox.information(self, "Modelo Guardado", f"El modelo ha sido guardado en {file_path}.")
        else:
            QMessageBox.warning(self, "Error", "Primero debes entrenar el modelo.")

class ConfiguraCapas(QMainWindow):
        def __init__(self, num_capas, parent = None):
            super(ConfiguraCapas, self).__init__(parent)
            self.setWindowTitle("Configurar Capas")
            self.num_capas = num_capas
            self.parent = parent  #guardar la referencia a la ventana principal
            self.setGeometry(400, 400, 350, 300)  
            self.initUI()
        def initUI(self):
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.layout = QVBoxLayout(self.central_widget)

            self.capas = []
            for i in range(self.num_capas):
                capa_label = QLabel(f"Capa {i + 1}")
                capa_neuronas = QLineEdit()
                capa_neuronas.setPlaceholderText("Número de Neuronas")
                capa_activacion = QComboBox()
                capa_activacion.addItems(["relu", "tanh", "softmax", "sigmoid", "linear"])
                self.layout.addWidget(capa_label)
                self.layout.addWidget(capa_neuronas)
                self.layout.addWidget(capa_activacion)
                self.capas.append([capa_neuronas, capa_activacion])

            self.submit_button = QPushButton("OK")
            self.submit_button.clicked.connect(self.guardar_capas)
            self.layout.addWidget(self.submit_button)

        def guardar_capas(self):
            capas_config = []
            for capa in self.capas:
                neuronas = int(capa[0].text())
                activacion = capa[1].currentText()
                capas_config.append((neuronas, activacion))
            self.parent.build_model(capas_config)
            self.close()
