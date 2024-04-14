# Instructions for compilation


Install right version of bazel:

```bash
wget https://releases.bazel.build/3.7.2/release/bazel-3.7.2-installer-linux-x86_64.sh
chmod +x bazel-3.7.2-installer-linux-x86_64.sh
sudo ./bazel-3.7.2-installer-linux-x86_64.sh
source /usr/local/lib/bazel/bin/bazel-complete.bash
```

Install python dependencies:

```
pip install numpy==1.19.5 
pip install keras_preprocessing==1.1.2
```

Build package tool (the actual tensorflow):

```bash
bazel build //tensorflow/tools/pip_package:build_pip_package
```

Build the python wheel
NOTE: wheel will be placed in the <dst-folder> folder, change it to a valid path

```bash
./bazel-bin/tensorflow/tools/pip_package/build_pip_package <dst-folder>
```

# After compilation

Install python wheel (move to another folder or create a virtual environment )

```bash
python3 -m venv tf_env
source tf_env/bin/activate
pip install protobuf==3.20.1
pip install <wheel-file>
```

Use tensorposit in python scripts:

```python
# This may generate some warnings, you can ignore it
import tensorflow as tf
from tensorflow.python import keras

# Set default floatx
keras.backend.set_floatx("posit160")

# Or create a tensor of posits
tp16e2 = tf.ones((10,10), dtype=tf.posit16e2)
tp8e2 = tf.ones((10,10), dtype=tf.posit8e2)
tp32e2 = tf.ones((10,10), dtype=tf.posit32e2)
```

