import onnx
from onnx import version_converter, helper

# import onnxruntime
# help(onnx)

# Preprocessing: load the model to be converted.
model_path = 'best1new.onnx'
original_model = onnx.load(model_path)
print(original_model.opset_import[0].version)
# print(original_model.Version)


# print('The model before conversion:\n{}'.format(original_model))
