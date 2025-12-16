# 💻  Otimização Avançada em Deep Learning com PyTorch: Uma Análise aplicada ao Fashion-MNIST

Este repositório contém o código, experimentos e visualizações desenvolvidas para a **Nota Técnica Aprofundada** do Capítulo 6 do livro *Deep Learning with PyTorch Step-by-Step*. O projeto aplica os conceitos de otimização de redes neurais ao modelo CNN LeNet-like treinado no popular dataset **Fashion-MNIST**.

---

## 👩‍💻 Autores
* **Julia Alanne Silvino dos Santos**
* **Pablo Durkheim Fernandes do Nascimento**

Este trabalho foi desenvolvido como projeto final da disciplina de **PROJETO DE SISTEMAS BASEADOS EM APRENDIZADO DE MÁQUINA**.

## 📄 Nota Técnica e Objetivos

Você pode conferir a análise detalhada, resultados e conclusões na nota técnica publicada:

> **🔗 Link para o Artigo no Medium:**
> [`Otimização Avançada em Deep Learning com PyTorch`](https://medium.com/@juliaalanne/otimiza%C3%A7%C3%A3o-avan%C3%A7ada-em-deep-learning-com-pytorch-uma-an%C3%A1lisem-aplicada-ao-fashion-mnist-8a0a7aa1095f?postPublishedType=repub)

### 🎯 Objetivos Técnicos
O projeto tem como objetivo:

* Demonstrar a função da **EWMA (Exponentially Weighted Moving Average)** no suavizamento de gradientes e o papel da **Correção de Viés (Bias Correction)**.
* Comparar o desempenho e a trajetória do **SGD Simples**, **SGD com Momentum** e **SGD com Nesterov**.
* Visualizar os **Gradientes Adaptados do Adam** (Primeiro e Segundo Momento) em um gráfico de três painéis.
* Implementar e comparar diferentes **Learning Rate Schedulers** ($\text{StepLR}$ e $\text{LambdaLR}$) e analisar seu impacto em treinamentos curtos.

---
