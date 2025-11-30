import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit import transpile
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


# 1️⃣ Gerar dataset simples de classificação
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_classes=2,
    n_clusters_per_class=2,
    n_redundant=0
)

# Normalizamos de 0 a pi porque os circuitos aceitam esse range
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=12
)

# 2️⃣ Criar Feature Map e Ansatz (circuito variacional)
feature_map = ZZFeatureMap(2)
ansatz = RealAmplitudes(2, reps=2)

# Circuito final = feature map + ansatz
circuit = feature_map.compose(ansatz)

# 3️⃣ Usamos o backend simulador local padrão do Qiskit 
estimator = Estimator()

# 4️⃣ Criar a rede neural quântica
qnn = EstimatorQNN(
    circuit=circuit,
    estimator=estimator,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

# 5️⃣ Otimizador COBYLA 
optimizer = COBYLA(maxiter=500)

# 6️⃣ Classificador baseado no QNN
classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer)

print("🟣 Treinando a QNN ")
classifier.fit(X_train, y_train)

# 7️⃣ Avaliando o modelo
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Acurácia final: {acc * 100:.2f}%")

# 8️⃣ Plotando a classificação
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
plt.title(f"Classificação com Rede Neural Quântica (QNN)\nAcurácia: {acc*100:.2f}%")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

