#!/bin/bash
echo "Removing models"
rm -rf models
mkdir out/
echo "Cloning models repo..."
# clone the tensorflow models on the colab cloud vm
git clone --q https://github.com/tensorflow/models.git
cd models/research

echo "Compile protoc"
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .

echo "Set up dependencies"
python -m pip install .

echo "Test compiled model builder"
python object_detection/builders/model_builder_tf2_test.py
