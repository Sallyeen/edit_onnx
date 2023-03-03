import onnx

input_path = 'resnet50_224_sim.onnx'
output_path = 'resnet50_extract.onnx'
input_names = ['flatten_473']
output_names = ['resnetv17_dense0_fwd']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)