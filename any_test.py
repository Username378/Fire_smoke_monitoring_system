import json

# 从文件中读取JSON数据
with open(r'E:\application\Security_monitoring_system\config\config.json', 'r') as f:
    data = json.load(f)

print(type(data['iou']))
