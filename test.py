import torch
from Net import Net
import numpy as np
from param import *
import matplotlib.pyplot as plt

model_name = "net64_32_ffb1f.pth"

# Загрузите веса модели
model_path = f"./models/{model_name}"
net = torch.load(model_path)
net.eval()  # Перевод в режим тестирования

net = net.cpu()
# Точное решение
def exact(x, t):
    return C * np.e ** (x + a**2 * t) 

x=np.linspace(0, l + 0.0, n + 1)
t=np.linspace(0, T + 0.0, m + 1)
ms_x, ms_t = np.meshgrid(x, t)

x = np.ravel(ms_x).reshape(-1,1)
t = np.ravel(ms_t).reshape(-1,1)


u = exact(x, t)
ms_u = u.reshape(ms_x.shape)

pt_x = torch.from_numpy(x).float()
pt_t = torch.from_numpy(t).float()
pt_y = net(pt_x, pt_t)
y = pt_y.data
ms_y = y.reshape(ms_x.shape)



# Точное решение
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].set_title("Exact Solution")
im1 = axes[0].imshow(ms_u, extent=[0, l, 0, T], origin='lower', aspect='auto', cmap='coolwarm')
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel(r"$t$")
fig.colorbar(im1, ax=axes[0], shrink=0.7, label=r"$u(x, t)$")

# Приближенное решение
axes[1].set_title("PINN Solution")
im2 = axes[1].imshow(ms_y, extent=[0, l, 0, T], origin='lower', aspect='auto', cmap='coolwarm')
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel(r"$t$")
fig.colorbar(im2, ax=axes[1], shrink=0.7, label=r"$u(x, t)$")

plt.tight_layout()

# Вычисляем точное решение
u_exact = ms_u.ravel()  # Точное решение (плоский массив)
u_pred = ms_y.numpy().ravel()   # Приближенное решение (плоский массив)

# L2-норма
l2_norm = np.sqrt(np.mean((u_exact - u_pred) ** 2))
# L∞-норма
linf_norm = np.max(np.abs(u_exact - u_pred))

print(f"L2-норма: {l2_norm:.6f}")
print(f"L∞-норма: {linf_norm:.6f}")

plt.show()