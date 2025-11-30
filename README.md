# 🧠 Quantum Neural Network (QNN) — Classificação Binária com Qiskit

Este projeto implementa um modelo de Rede Neural Quântica (QNN) utilizando o framework Qiskit Machine Learning para resolver um problema simples de classificação binária.
O objetivo é demonstrar como circuitos quânticos podem ser aplicados em Machine Learning e comparar sua performance com métodos clássicos.

## 📌 📁 Estrutura do projeto
```
📦 Projeto-QNN
 ┣ 📜 qnn_classifier.py     # Código principal da QNN
 ┗ 📜 README.md             # Documentação do projeto
```

## 🧬 Objetivo do Projeto

O objetivo é treinar uma rede neural quântica para distinguir dois grupos de dados usando:
* Feature Map (ZZFeatureMap) para codificação dos dados no circuito quântico

* Ansatz variacional (RealAmplitudes)
* EstimatorQNN como arquitetura da neural network
* Otimizador COBYLA
* Simulador quântico padrão do Qiskit

O modelo é treinado usando um dataset gerado artificialmente.

## 📊 Como o modelo funciona

### 1. Geração do dataset
* Um conjunto de dados sintéticos é criado com duas features e duas classes.

### 2. Normalização para o intervalo 0 → π
* Os circuitos do Qiskit utilizam rotações, então mapeamos os dados para esse intervalo.

### 3. Construção do circuito quântico

* ZZFeatureMap: codifica os dados nos qubits

* RealAmplitudes: insere parâmetros treináveis

* A composição dos dois define a arquitetura final da rede quântica

### 4. Treinamento com COBYLA
* O otimizador ajusta os parâmetros do circuito para minimizar a função de perda.

### 5. Classificação e avaliação
* Após treinado, o modelo prevê as classes no conjunto de teste, e exibimos:
```
 Acurácia
 Gráfico de classificação
```

## 📈 Exemplo de saída esperada

* Acurácia entre 40% e 75% (normal para QNNs pequenas)

* Gráfico colorido mostrando a separação das classes prevista pela QNN

* Impressão do processo de treinamento no console

## ⚠️ Sobre a Acurácia

Modelos de Machine Learning quânticos ainda são sensíveis a:

* Ruído

* Tamanho pequeno do circuito

* Dataset simples

* Simulador idealizado

Por isso a acurácia pode ficar abaixo de modelos clássicos.
Mas isso não é um problema:
O objetivo é demonstrar a aplicação de computação quântica em ML, não competir com modelos clássicos.

## 🚀 Como rodar o projeto
### 1. Clone o repositório
```
git clone https://github.com/seuusuario/Projeto-QNN.git
```
### 2. Crie um ambiente virtual
```
python -m venv venv
```
### 3. Ative o venv
```
venv\Scripts\activate
```
### 4. Instale as dependências
```
pip install qiskit-machine-learning qiskit scikit-learn matplotlib
```
### 5. Execute o script
```
python qnn_classifier.py
```
## 📚 Tecnologias utilizadas
```
Python 3

Qiskit

Qiskit Machine Learning

Scikit-Learn

Matplotlib
```
## 🧑‍💻 Autor

Projeto desenvolvido por Lucas Ribeiro, como parte de estudos em Computação Quântica Aplicada à Inteligência Artificial.
