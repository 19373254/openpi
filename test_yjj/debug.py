import jax
import numpy as np

self_structure = {'image_110': 'image_0', 'state': 'observation.state', 'action': 'action'}
flat_item = {
    'image_0': [0.5, 0.2], 
    'observation.state': [0.5, 0.2], 
    'action': [1.0, -0.5]
}

# 执行映射
result = jax.tree.map(lambda k: flat_item[k], self_structure)

print(result)
# # 输出结果
# {
#     'image_0': flat_item['image_0'],  # 对应 np.array(...)
#     'observation.state': flat_item['observation.state'],  # 对应 [0.5, 0.2]
#     'action': flat_item['action']  # 对应 [1.0, -0.5]
# }

