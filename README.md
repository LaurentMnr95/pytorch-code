# pytorch-code
## To train the model

nohup visdom -logging_level 100  >output_visdom.log & (to look to the training progress)

python train_classifier.py

The options are modifiable in options_classifier.py (choose for models, epochs etc.)


## To test the model

python test_classifier.py