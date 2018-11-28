## Script for downloading data

# GloVe Vectors
wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip

# Questions
wget -P data http://visualqa.org/data/abstract_v002/vqa/Questions_Train_abstract_v002.zip
unzip data/Questions_Train_abstract_v002.zip -d data
rm data/Questions_Train_abstract_v002.zip

wget -P data http://visualqa.org/data/abstract_v002/vqa/Questions_Val_abstract_v002.zip
unzip data/Questions_Val_abstract_v002.zip -d data
rm data/Questions_Val_abstract_v002.zip

wget -P data http://visualqa.org/data/abstract_v002/vqa/Questions_Test_abstract_v002.zip
unzip data/Questions_Test_abstract_v002.zip -d data
rm data/Questions_Test_abstract_v002.zip

# Annotations
wget -P data http://visualqa.org/data/abstract_v002/vqa/Annotations_Train_abstract_v002.zip
unzip data/Annotations_Train_abstract_v002.zip -d data
rm data/Annotations_Train_abstract_v002.zip

wget -P data http://visualqa.org/data/abstract_v002/vqa/Annotations_Val_abstract_v002.zip
unzip data/Annotations_Val_abstract_v002.zip -d data
rm data/Annotations_Val_abstract_v002.zip

# Image Features
# wget -P data https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
# unzip data/trainval_36.zip -d data
# rm data/trainval_36.zip
