import tensorflow as tf

print(tf.test.is_built_with_cuda())  # Check if TensorFlow is built with CUDA support
print(tf.test.is_built_with_tensorrt())  # Check if TensorFlow is built with TensorRT support

