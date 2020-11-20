import numpy as np
from lib.yamllib import read_yaml

class Parameter():
  def __init__(self, tensor):
    self.tensor = tensor
    self.gradient = np.zeros_like(self.tensor)

class Layer:
  def __init__(self):
    self.parameters = []

  def forward(self, X):
    return X, lambda D: D

  def build_param(self, tensor):
    param = Parameter(tensor)
    self.parameters.append(param)
    return param

  def update(self, optimizer):
    for param in self.parameters: optimizer.update(param)

class Linear(Layer):
  def __init__(self, inputs, outputs, value_range, value_diff):
    super().__init__()
    #self.weights = self.build_param(np.random.randn(inputs, outputs) * np.sqrt(1 / inputs))
    self.weights = self.build_param((value_range * np.random.random((inputs, outputs)) + value_diff) * np.sqrt(1 / inputs))
    self.bias = self.build_param(np.zeros(outputs))
    
  def forward(self, X):
    def backward(D):
      self.weights.gradient += X.T @ D
      self.bias.gradient += D.sum(axis=0)
      return D @ self.weights.tensor.T
    return X @ self.weights.tensor + self.bias.tensor, backward
  
class Sequential(Layer):
  def __init__(self, *layers):
    super().__init__()
    self.layers = layers
    for layer in layers:
      self.parameters.extend(layer.parameters)
    
  def forward(self, X):
    backprops = []
    Y = X
    for layer in self.layers:
      Y, backprop = layer.forward(Y)
      backprops.append(backprop)
    def backward(D):
      for backprop in reversed(backprops):
        D = backprop(D)
      return D
    return Y, backward


class ReLu(Layer):
  def forward(self, X):
    mask = X > 0
    return X * mask, lambda D: D * mask
  
class Sigmoid(Layer):
  def forward(self, X):
    S = 1 / (1 + np.exp(-X))
    def backward(D):
      return D * S * (1 - S)
    return S, backward

def mse_loss(Y_, Y):
  diff = Y_ - Y.reshape(Y_.shape)
  return np.square(diff).mean(), 2 * diff / len(diff)
  
def ce_loss(Y_, Y):
  num = np.exp(Y_)
  den = num.sum(axis=1).reshape(-1, 1)
  prob = num / den
  log_den = np.log(den)
  ce = np.inner(Y_ - log_den, Y)
  return ce.mean(), Y - prob / len(Y)


class SGDOptimizer():
  def __init__(self, lr=0.1):
    self.lr = lr

  def update(self, param):
    param.tensor -= self.lr * param.gradient
    param.gradient.fill(0)


class Learner():
  def __init__(self, model, loss, optimizer):
    self.model = model
    self.loss = loss
    self.optimizer = optimizer
      
  def fit_batch(self, X, Y):
    Y_, backward = self.model.forward(X)
    L, D = self.loss(Y_, Y)
    backward(D)
    self.model.update(self.optimizer)
    return L
    
  def fit(self, X, Y, epochs, bs):
    losses = []
    for epoch in range(epochs):
      p = np.random.permutation(len(X))
      L = 0
      for i in range(0, len(X), bs):
        X_batch = X[p[i:i + bs]]
        Y_batch = Y[p[i:i + bs]]
        L += self.fit_batch(X_batch, Y_batch)
      losses.append(L)
    return losses

class Lens():
  def __init__(self, dataset_shape):
    self.num_samples, self.num_features = dataset_shape
    self.config = read_yaml("lib/lensconf.yaml")
    self.epochs        = self.config["epochs"]
    self.batch_size    = self.config["batch_size"]
    self.learning_rate = self.config["learning_rate"]
    self.value_range   = self.config["value_range"]
    self.value_diff    = self.config["value_diff"]
    
    self.linear = Linear(self.num_features, 1, self.value_range, self.value_diff)
    self.model = Sequential(self.linear)
    self.learner = Learner(self.model, mse_loss, SGDOptimizer(lr=self.learning_rate))
    
  def learn(self, X, Y):
    print(self.linear.weights.tensor)
    print(self.linear.bias.tensor)
    return self.learner.fit(X, Y, epochs=self.epochs, bs=self.batch_size)
  
  def forward(self, X):
    Y, b = self.model.forward(X)
    return Y
    