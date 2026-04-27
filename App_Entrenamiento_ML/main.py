import sys
import io
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # Configurar la salida de texto para usar UTF-8
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidgetItem, QFileDialog, QMessageBox
from mainwindow_ui import Ui_MainWindow
from NN_Model import NNWindow
from DT_Model import DTWindow
from KNN_Model import KNNWindow
from SVC_Model import SVCWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.selectDatasetPushButton.clicked.connect(self.importar_dataset)
        self.NNPushButton.clicked.connect(self.NNModel)
        self.KNNPushButton.clicked.connect(self.KNNModel) 
        self.DTPushButton.clicked.connect(self.DTModel)
        self.SVCPushButton.clicked.connect(self.SVCModel)
        self.dataset = None  
        self.filasTotal = 0

    def importar_dataset(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Seleccionar Dataset", "", "CSV Files (*.csv);;All Files (*)")
        if fileName:
            self.mostrar_dataset(fileName)

    def mostrar_dataset(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.datasetTableWidget.setRowCount(self.dataset.shape[0]) 
        self.datasetTableWidget.setColumnCount(self.dataset.shape[1]) 
        self.datasetTableWidget.setHorizontalHeaderLabels(self.dataset.columns) 

        for i in range(self.dataset.shape[0]):
            for j in range(self.dataset.shape[1]):
                self.datasetTableWidget.setItem(i, j, QTableWidgetItem(str(self.dataset.iat[i, j])))

        self.filasValueLabel.setText(str(self.dataset.shape[0]))
        self.columnasValueLabel.setText(str(self.dataset.shape[1]))

    #RED NEURONAL
    def NNModel(self):
        if self.dataset is not None:
            self.nn_window = NNWindow(self.dataset)
            self.nn_window.show()
        else:
            QMessageBox.warning(self, "Error", "Primero debes importar un dataset.")
    #K VECINOS MÁS CERCANOS
    def KNNModel(self):
        if self.dataset is not None:
            self.knn_window = KNNWindow(self.dataset)
            self.knn_window.show()
        else:
            QMessageBox.warning(self, "Error", "Primero debes importar un dataset.")
    #ÁRBOL DE DECISIÓN
    def DTModel(self):
        if self.dataset is not None:
            self.dt_window = DTWindow(self.dataset)
            self.dt_window.show()
        else:
            QMessageBox.warning(self, "Error", "Primero debes importar un dataset.")
    #MÁQUINA DE SOPORTE VECTORIAL PARA CLASIFICACIÓN
    def SVCModel(self):
        if self.dataset is not None:
            self.svc_window = SVCWindow(self.dataset)
            self.svc_window.show()
        else:
            QMessageBox.warning(self, "Error", "Primero debes importar un dataset.") 




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


