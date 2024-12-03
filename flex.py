import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")

# Параметры задачи
l = 2          # Длина стержня
T = 1          # Конечное время
n = 64         # Количество узлов длины стержня
m = 32         # Количество узлов времени T
h = l / n      # Шаг сетки по l
tau = T / m    # Шаг сетки по T

# Коэффициенты из задачи
C = 0.1        # Коэффициент C
a = 1          # Коэффициент a

# Начальные и граничные условия
def InitialCondition(x):
    # Начальное условие: u(x, t=0)
    return C * np.e ** (x) 

def BoundaryLeftCondition(t):
    # Левое граничное условие: u(x=0, t)
    return C * np.e ** (a**2 * t)

def BoundaryRightCondition(t):
    # Правое граничное условие: u(x=l, t)
    return C * np.e ** (l + (a * a) * t) 

def func(x):
    # Правая часть уравнения, в данном случае равна sin(x)
    return 0.0

def pde(x, t, net):
    # Определение уравнения в частных производных
    u = net(x, t)  # Сеточная функция u(x, t)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]  # Первая производная по x
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]  # Вторая производная по x
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]  # Первая производная по t
    pde = u_xx - u_t + func(x)  # Уравнение: u_xx - u_t + f(x)
    return pde

# Подготовка данных для обучения
x_bc = np.array([i * h for i in range(n)])  # Сетка по x для граничных условий
t_bc = np.array([i * tau for i in range(m)])  # Сетка по t для граничных условий

# Формирование граничных условий
bc_l = np.vstack([np.zeros((m)),    t_bc,  BoundaryLeftCondition(t_bc)]).T  # Левое граничное условие
bc_r = np.vstack([np.ones((m)) * l, t_bc,  BoundaryRightCondition(t_bc)]).T  # Правое граничное условие
ic   = np.vstack([x_bc,     np.zeros((n)), InitialCondition(x_bc)]).T  # Начальное условие

# Объединение начальных и граничных условий
initial_boundary_conditions = np.vstack([bc_l, bc_r, ic])
x_initial_bc = initial_boundary_conditions[:, 0].reshape(-1, 1)  # Координаты x для начальных и граничных условий
t_initial_bc = initial_boundary_conditions[:, 1].reshape(-1, 1)  # Координаты t для начальных и граничных условий
u_initial_bc = initial_boundary_conditions[:, 2].reshape(-1, 1)  # Значения u для начальных и граничных условий

# Преобразование данных в тензоры
pt_x_initial_bc = torch.autograd.Variable(torch.from_numpy(x_initial_bc).float(), requires_grad=False).to(device)
pt_t_initial_bc = torch.autograd.Variable(torch.from_numpy(t_initial_bc).float(), requires_grad=False).to(device)
pt_u_initial_bc = torch.autograd.Variable(torch.from_numpy(u_initial_bc).float(), requires_grad=False).to(device)


# Создание структурированной сетки для внутренних узлов
x_collocation = np.linspace(0, l, n+1)[1:-1]  # Узлы по x без учета граничных точек
t_collocation = np.linspace(0, T, m+1)[1:]    # Узлы по t без начальной точки

# Создание структурированной сетки
X_struct, T_struct = np.meshgrid(x_collocation, t_collocation)

# Преобразование двумерной сетки в список координат 
x_collocation_reshaped = X_struct.flatten().reshape(-1, 1)  # Все координаты x
t_collocation_reshaped = T_struct.flatten().reshape(-1, 1)  # Все координаты t

# Конвертация в PyTorch тензоры
pt_x_collocation = torch.autograd.Variable(torch.from_numpy(x_collocation_reshaped).float(), requires_grad=True).to(device)
pt_t_collocation = torch.autograd.Variable(torch.from_numpy(t_collocation_reshaped).float(), requires_grad=True).to(device)

import hashlib
import json
from datetime import datetime

def Save(net, num_layer,num_per_layer, activation, learning_rate, weight_IBC, weight_PDE, MAX_EPOCHS, loss_IBC, loss_PDE, loss_COMM):
    # Сохранение модели
    net = net.cpu()
    
    # Функция для генерации хэша
    def generate_hash(data, length=5):
        hash_object = hashlib.md5(str(data).encode())
        return hash_object.hexdigest()[:length]

    # Создаем метаданные для хэша
    hash_data = {
        "final_loss": {
            "total_loss": loss_COMM[-1].item(),
            "ibc_loss": loss_IBC[-1].item(),
            "pde_loss": loss_PDE[-1].item()
        },
    }

    # Генерируем хэш
    model_hash = generate_hash(hash_data)
    model_filename = f'net{n}_{m}_{model_hash}.pth'

    torch.save(net, model_filename)
    end_time = time.time()
    # Сохранение метаданных
    import os

    # Настройка имени файлов
    metadata_filename = f'metadata.json'

    # Функция для загрузки существующих метаданных
    def load_metadata(filename):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        return []

    # Загрузка существующих метаданных
    metadata_list = load_metadata(metadata_filename)

    # Новые метаданные
    new_metadata = {
        "model_name": model_filename,
        "final_loss": {
            "total_loss": loss_COMM[-1].item(),
            "ibc_loss": loss_IBC[-1].item(),
            "pde_loss": loss_PDE[-1].item()
        },
        "hyperparameters": {
            "optimizer": "LBFGS",
            "optimizer_lr": learning_rate,
            "max_epochs_lbfgs": MAX_EPOCHS,
            "weight_PDE": weight_PDE,
            "weight_IBC": weight_IBC
        },
        "Net": {
            "num_layers":num_layer,
            "num_per_layers": num_per_layer,
            "activation" : activation,
        }
    }

    # Добавляем новые метаданные в список
    metadata_list.append(new_metadata)

    # Сохранение обновленных метаданных
    with open(metadata_filename, "w") as f:
        json.dump(metadata_list, f, indent=4)

    print(f"Модель сохранена в {model_filename}")
    print(f"Обновленные метаданные сохранены в {metadata_filename}")


# @title Архитектура нейронной сети
class Net(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, mid_features * 1))
        self.layers.append(nn.Linear(mid_features * 1, mid_features * 2))
        self.layers.append(nn.Linear(mid_features * 2, mid_features * 2))
        self.layers.append(nn.Linear(mid_features * 2, out_features))
        self.activation = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        x = inputs
        for layer in self.layers[:-1]:  # Проходим через все, кроме последнего слоя
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
    

import random

MAX_EPOCHS = 45

mse_cost_function = torch.nn.MSELoss()

# Диапазоны для гиперпараметров
hyperparameter_space = {
    "neurons_per_layer": [5, 6, 7],  # Число нейронов на слой
    "learning_rate": [0.05],  # Скорость обучения
    "weight_PDE": [0.9, 1.0, 1.1],  # Вес потерь PDE
    "weight_IBC": [0.9, 1.0, 1.1],  # Вес потерь IBC
    "activation": [0],
}
num_layer = 4
num_per_layer = 4
learning_rate = 0.01
activation = 0
# Случайный поиск
def random_search(n_iterations):
    global num_layer, num_per_layer, learning_rate, activation
    best_loss = float("inf")
    best_params = None
    print("Start Finding")
    for iteration in range(n_iterations):
        # Случайное сочетание гиперпараметров
        params = {key: random.choice(values) for key, values in hyperparameter_space.items()}
        
        # Настройка параметров
        global weight_PDE, weight_IBC
        weight_PDE = params["weight_PDE"]
        weight_IBC = params["weight_IBC"]

        # Настройка сети
        net = Net(2,params["neurons_per_layer"],1).to(device)
        optimizer = torch.optim.LBFGS(net.parameters(), lr=params["learning_rate"])
        num_layer = 4
        num_per_layer = [params["neurons_per_layer"]] * num_layer
        learning_rate = params["learning_rate"]
        activation = params["activation"]
        # Тренировка и получение финальной ошибки
        final_loss = train_and_evaluate(net, optimizer)  # Нужно реализовать функцию train_and_evaluate

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params
        print(f"iter = {iteration} ")
    return best_params, best_loss

# Функция для обучения и получения финальной ошибки
def train_and_evaluate(net, optimizer):
    def get_loss(pt_x_collocation, pt_t_collocation, batch_size=None):
        net_bc_ic_out = net(pt_x_initial_bc, pt_t_initial_bc)
        mse_bc_ic_u = mse_cost_function(net_bc_ic_out, pt_u_initial_bc)

        all_zeros = np.zeros((n,1))
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

        # PDE потери для всей сетки или батчами
        mse_pde = 0
        if batch_size:
            # Разбиваем на батчи
            num_points = pt_x_collocation.shape[0]
            for i in tqdm(range(0, num_points, batch_size), desc="Collocation Loss", leave=False):
                x_batch = pt_x_collocation[i:i + batch_size]
                t_batch = pt_t_collocation[i:i + batch_size]
                pde_out = pde(x_batch, t_batch, net)
                mse_pde += mse_cost_function(pde_out, torch.zeros_like(pde_out))
        else:
            # Полная обработка сетки
            pde_out = pde(pt_x_collocation, pt_t_collocation, net)
            mse_pde = mse_cost_function(pde_out, torch.zeros_like(pde_out))

        return mse_bc_ic_u , mse_pde
    loss_IBC = np.zeros((MAX_EPOCHS, 1))
    loss_PDE = np.zeros((MAX_EPOCHS, 1))
    loss_COMM = np.zeros((MAX_EPOCHS, 1))
    pbar = tqdm(range(MAX_EPOCHS), desc="Training", leave=True)
    for epoch in pbar:
        # Closure для LBFGS
        def closure():
            loss_ibc, loss_pde = get_loss(pt_x_collocation, pt_t_collocation)
            loss = loss_ibc * weight_IBC + loss_pde * weight_PDE
            optimizer.zero_grad()
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        loss_ibc, loss_pde = get_loss(pt_x_collocation, pt_t_collocation, batch_size=None)
        
        loss = loss_ibc * weight_IBC + loss_pde * weight_PDE

        loss_COMM[epoch] = (loss.detach().cpu().numpy().item())
        loss_IBC[epoch] = (loss_ibc.detach().cpu().numpy().item())
        loss_PDE[epoch] = (loss_pde.detach().cpu().numpy().item())
        pbar.set_description(f"Epoch {epoch+1}/{MAX_EPOCHS} | Loss: {loss:.10f}")
    Save(net, num_layer,num_per_layer, activation, learning_rate, weight_IBC, weight_PDE, MAX_EPOCHS, loss_IBC, loss_PDE, loss_COMM)
    return loss_COMM[-1]  # Финальная ошибка

# Запуск подбора
if __name__ == "__main__":

    best_params, best_loss = random_search(n_iterations=20)
    print("Лучшие параметры:", best_params)
    print("Лучший loss:", best_loss)
