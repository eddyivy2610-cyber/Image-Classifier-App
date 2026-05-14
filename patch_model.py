import h5py
import json

def remove_quant(obj):
    if isinstance(obj, dict):
        if 'quantization_config' in obj:
            del obj['quantization_config']
        for k, v in obj.items():
            remove_quant(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_quant(item)

with h5py.File('backend/model/cifar100_efficientnet.h5', 'r+') as f:
    model_config_str = f.attrs.get('model_config')
    if model_config_str is None:
        print("No model config found.")
    else:
        model_config = json.loads(model_config_str)
        remove_quant(model_config)
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
        print("Patched successfully")
