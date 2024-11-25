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
n = 32         # Количество узлов длины стержня
m = 16         # Количество узлов времени T
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
    # Правая часть уравнения, в данном случае равна 0
    return 0

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

# Преобразование двумерной сетки в список координат (для подачи в нейросеть)
x_collocation_reshaped = X_struct.flatten().reshape(-1, 1)  # Все координаты x
t_collocation_reshaped = T_struct.flatten().reshape(-1, 1)  # Все координаты t

# Конвертация в PyTorch тензоры
pt_x_collocation = torch.autograd.Variable(torch.from_numpy(x_collocation_reshaped).float(), requires_grad=True).to(device)
pt_t_collocation = torch.autograd.Variable(torch.from_numpy(t_collocation_reshaped).float(), requires_grad=True).to(device)


class Net(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super(Net, self).__init__()
        # Инициализация слоев сети с различными размерами
        self.lin_linear_1 = self.make_layer(in_features, mid_features * 1)  # Первый линейный слой
        self.lin_linear_2 = self.make_layer(mid_features * 1, mid_features * 2)  # Второй линейный слой
        self.lin_linear_3 = self.make_layer(mid_features * 2, mid_features * 2)  # Третий линейный слой
        self.lin_linear_4 = self.make_layer(mid_features * 2, out_features)  # Четвертый линейный слой (выходной)
        self.tanh = F.tanh  # Функция активации гиперболический тангенс

    # Метод для создания линейного слоя
    def make_layer(self, in_f, out_f):
        return nn.Linear(in_f, out_f)

    # Прямой проход сети
    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)  # Объединение входных данных x и t вдоль оси 1
        x = self.lin_linear_1(inputs)  # Применение первого линейного слоя
        x = self.tanh(x)  # Применение функции активации
        x = self.lin_linear_2(x)  # Применение второго линейного слоя
        x = self.tanh(x)  # Применение функции активации
        x = self.lin_linear_3(x)  # Применение третьего линейного слоя
        x = self.tanh(x)  # Применение функции активации
        x = self.lin_linear_4(x)  # Применение четвертого линейного слоя (выходного)
        return x  # Возвращаем выходные данные


# Настройка оптимизатора и нейросети, loss MSE
MAX_EPOCHS = 100  # Максимальное количество эпох
net = Net(2, 6, 1)  # Создание сети с 2 входами, 6 нейронами в скрытых слоях и 1 выходом
net = net.to(device)  # Перенос модели на выбранное устройство (GPU или CPU)
mse_cost_function = torch.nn.MSELoss()  # Определение функции потерь (среднеквадратичная ошибка)
optimizer = torch.optim.LBFGS(list(net.parameters()), lr=0.05)  # Оптимизатор LBFGS с шагом обучения 0.05
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Альтернативный оптимизатор Adam

# @title Training / Fitting
def get_loss(pt_x_collocation, pt_t_collocation, batch_size=None):
    # Вычисление потерь для начальных/граничных условий
    net_bc_ic_out = net(pt_x_initial_bc, pt_t_initial_bc)
    mse_bc_ic_u = mse_cost_function(net_bc_ic_out, pt_u_initial_bc)

    # Для вычисления потерь PDE, создаем массив нулей
    all_zeros = np.zeros((n, 1))
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    # Потери для физического уравнения (PDE)
    mse_pde = 0
    if batch_size:
        # Если задан размер батча, обрабатываем данные батчами
        num_points = pt_x_collocation.shape[0]
        for i in range(0, num_points, batch_size):
            x_batch = pt_x_collocation[i:i + batch_size]  # Выбор батча данных по x
            t_batch = pt_t_collocation[i:i + batch_size]  # Выбор батча данных по t
            pde_out = pde(x_batch, t_batch, net)  # Вычисление предсказаний по PDE
            mse_pde += mse_cost_function(pde_out, torch.zeros_like(pde_out))  # Вычисление потерь для PDE
    else:
        # Если не используется батчинг, обрабатываем все данные сразу
        pde_out = pde(pt_x_collocation, pt_t_collocation, net)
        mse_pde = mse_cost_function(pde_out, torch.zeros_like(pde_out))

    return mse_bc_ic_u + mse_pde  # Общие потери (сумма потерь для начальных/граничных условий и PDE)

losses = np.zeros((MAX_EPOCHS, 1))  # Массив для хранения значений потерь на каждой эпохе
pbar = tqdm(range(MAX_EPOCHS), desc="Training", leave=True)  # Прогресс-бар для обучения
start_time = time.time()  # Засекаем время начала обучения
i = 0
for epoch in pbar:
    net.train()  # Переводим модель в режим обучения

    # Определение closure, который используется для вычисления потерь и градиентов
    def closure():
        loss = get_loss(pt_x_collocation, pt_t_collocation)  # Вычисляем потери
        optimizer.zero_grad()  # Обнуляем градиенты перед вычислением новых
        loss.backward()  # Обратное распространение ошибок (вычисление градиентов)
        return loss  # Возвращаем потери для шага оптимизации

    # Шаг оптимизатора: обновление весов сети
    optimizer.step(closure)

    # Вычисление потерь на текущей эпохе для сохранения в график
    loss = get_loss(pt_x_collocation, pt_t_collocation, batch_size=None)
    losses[i] = (loss.detach().cpu().numpy().item())  # Сохраняем значение потерь
    i += 1  # Увеличиваем индекс для следующего сохранения потерь

    # Обновление прогресс-бара с текущими значениями
    pbar.set_description(f"Epoch {epoch + 1}/{MAX_EPOCHS} | Loss: {loss:.10f}")

print("--- %s seconds ---" % (time.time() - start_time))  # Время завершения обучения

# @title График Loss
# Построение графика потерь по эпохам
plt.plot(np.arange(0, MAX_EPOCHS, 1), losses, label="loss")
plt.grid()  # Добавление сетки на график
plt.xlabel = "Epochs"  # Подпись оси X
plt.ylabel = "losses"  # Подпись оси Y
plt.legend()  # Добавление легенды



# @title Точное
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def exact(x, t):
    return C * np.e ** (x + (a * a) * t )

x=np.linspace(0, l, n+1)
t=np.linspace(0, T, m+1)
ms_x, ms_t = np.meshgrid(x, t)

x = np.ravel(ms_x).reshape(-1,1)
t = np.ravel(ms_t).reshape(-1,1)


u = exact(x, t)
ms_u = u.reshape(ms_x.shape)


x=np.linspace(0, l, n+1)
t=np.linspace(0, T, m+1)
ms_x, ms_t = np.meshgrid(x, t)

x = np.ravel(ms_x).reshape(-1,1)
t = np.ravel(ms_t).reshape(-1,1)

pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
pt_y = net(pt_x, pt_t)
y = pt_y.data.cpu().numpy()
ms_y = y.reshape(ms_x.shape)




fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].set_title("Exact Solution")
im1 = axes[0].imshow(ms_u, extent=[0, l, 0, T], origin='lower', aspect='auto', cmap='coolwarm')
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel(r"$t$")
fig.colorbar(im1, ax=axes[0], shrink=0.7, label=r"$u(x, t)$")

# Приближенное решение
axes[1].set_title("Approximate Solution")
im2 = axes[1].imshow(ms_y, extent=[0, l, 0, T], origin='lower', aspect='auto', cmap='coolwarm')
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel(r"$t$")
fig.colorbar(im2, ax=axes[1], shrink=0.7, label=r"$u(x, t)$")

plt.tight_layout()



import numpy as np

# Вычисляем точное решение
u_exact = ms_u.ravel()  # Точное решение (плоский массив)
u_pred = ms_y.ravel()   # Приближенное решение (плоский массив)

# L2-норма
l2_norm = np.sqrt(np.mean((u_exact - u_pred) ** 2))

# L∞-норма
linf_norm = np.max(np.abs(u_exact - u_pred))

print(f"L2-норма: {l2_norm:.6f}")
print(f"L∞-норма: {linf_norm:.6f}")


plt.show()