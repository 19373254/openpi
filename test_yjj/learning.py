# import jax
# import jax.numpy as jnp
# from jax import grad, jit, pmap
# import optax  # 常用的优化库

# # 创建一个简单的模型
# def model(params, x):
#     return jnp.dot(x, params)

# # 损失函数
# def loss(params, x, y):
#     preds = model(params, x)
#     return jnp.mean((preds - y) ** 2)

# # 更新函数
# @jit
# def update(params, x, y, opt_state, optimizer):
#     grads = grad(loss)(params, x, y)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     new_params = optax.apply_updates(params, updates)
#     return new_params, opt_state

# # 并行训练函数
# @pmap
# def train_step(params, x, y, opt_state, optimizer):
#     return update(params, x, y, opt_state, optimizer)

# # 初始化参数和优化器
# num_devices = jax.device_count()
# params = jnp.ones((num_devices, 10))  # 假设每个设备的参数
# optimizer = optax.adam(learning_rate=0.01)
# opt_state = optimizer.init(params)

# # 模拟输入数据
# x_data = jnp.ones((num_devices, 5))  # 每个设备的输入
# y_data = jnp.ones((num_devices, 1))   # 每个设备的目标

# # 训练循环
# for epoch in range(10):
#     print(f"params shape: {params.shape}, x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")
#     params, opt_state = train_step(params, x_data, y_data, opt_state, optimizer)
#     print(f"Epoch {epoch + 1}: params = {params}")




import jax
import jax.numpy as jnp
from jax import grad, jit, pmap
import optax

# 模型定义
def model(params, x):
    return jnp.dot(x, params)

# 损失函数
def loss(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) **2)

# 更新函数（保持设备独立性）
@jit
def update(params, x, y, opt_state, optimizer):
    grads = grad(loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# 并行训练函数（通过pmap分片）
@pmap
def train_step(params, x, y, opt_state, optimizer):
    return update(params, x, y, opt_state, optimizer)

# 初始化参数和优化器
num_devices = jax.device_count()
params = jax.pmap(lambda: jnp.ones(10))()  # 每个设备独立参数
optimizer = optax.adam(0.01)
opt_state = jax.pmap(optimizer.init)(params)  # 分片优化器状态

# 数据生成（分片到设备）
x_data = jax.pmap(lambda: jnp.ones(5))()
y_data = jax.pmap(lambda: jnp.ones(1))()

# 训练循环
for epoch in range(10):
    params, opt_state = train_step(params, x_data, y_data, opt_state, optimizer)
    print(f"Epoch {epoch+1}: params = {params}")








