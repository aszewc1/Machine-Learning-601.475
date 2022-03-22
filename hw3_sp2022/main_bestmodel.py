import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (
    dev_acc_and_loss, accuracy, approx_train_acc_and_loss)


class BestModel(torch.nn.Module):
    # TODO Implement your model's structure and input/filter/output dimensions
    def __init__(self, hidden_layer_width, input_height, input_width, n_classes):
        super().__init__()

        self.height = input_height
        self.width = input_width

        self.conv1 = torch.nn.Conv2d(1, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.maxPool1 = torch.nn.MaxPool2d(2, stride=2)

        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.conv4 = torch.nn.Conv2d(64, 128, 3)
        self.maxPool2 = torch.nn.MaxPool2d(8, stride=8)

        self.lin1 = torch.nn.Linear(128, hidden_layer_width)
        self.lin2 = torch.nn.Linear(hidden_layer_width, n_classes)

    def forward(self, x):

        x = x.reshape((x.shape[0], 1, self.height, self.width))

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxPool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxPool2(x)

        x = x.squeeze(axis=2)
        x = x.squeeze(axis=2)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x


def normalize(img):
    flat = img.flatten()
    norm = (flat - np.mean(flat)) / np.std(flat)
    return norm


if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')

    if MODE == "train":

        LOG_DIR = arguments.get('log_dir')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        BATCH_SIZE = arguments.get('bs')
        EPOCHS = arguments.get('epochs')
        DATE_PREFIX = datetime.datetime.now().strftime('%Y%m%d%H%M')
        if LEARNING_RATE is None:
            raise TypeError("Learning rate has to be provided for train mode")
        if BATCH_SIZE is None:
            raise TypeError("batch size has to be provided for train mode")
        if EPOCHS is None:
            raise TypeError(
                "number of epochs has to be provided for train mode")
        TRAIN_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_images.npy"))
        TRAIN_LABELS = np.load(os.path.join(DATA_DIR, "fruit_labels.npy"))
        DEV_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_dev_images.npy"))
        DEV_LABELS = np.load(os.path.join(DATA_DIR, "fruit_dev_labels.npy"))

        # TODO format your dataset to the appropriate shape/dimensions necessary to be input into your model.

        N_IMAGES, HEIGHT, WIDTH = TRAIN_IMAGES.shape
        N_CLASSES = len(np.unique(TRAIN_LABELS))

        # TODO Normalize your dataset if desired

        flat_train_imgs = np.array([normalize(img) for img in TRAIN_IMAGES])
        flat_dev_imgs = np.array([normalize(img) for img in DEV_IMAGES])

        # do not touch the following 4 lines (these write logging model performance to an output file
        # stored in LOG_DIR with the prefix being the time the model was trained.)
        LOGFILE = open(os.path.join(LOG_DIR, f"bestmodel.log"), 'w')
        log_fieldnames = ['step', 'train_loss',
                          'train_acc', 'dev_loss', 'dev_acc']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()

        # TODO change depending on your model's instantiation
        model = BestModel(hidden_layer_width=200, input_height=HEIGHT,
                          input_width=WIDTH, n_classes=N_CLASSES)

        # TODO (OPTIONAL) : you can change the choice of optimizer here if you wish.
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for step in range(EPOCHS):
            i = np.random.choice(
                flat_train_imgs.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(flat_train_imgs[i].astype(np.float32))
            y = torch.from_numpy(TRAIN_LABELS[i].astype(np.int))

            # Forward pass: Get logits for x
            logits = model(x)
            # Compute loss
            loss = F.cross_entropy(logits, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log model performance every 100 epochs
            if step % 100 == 0:
                train_acc, train_loss = approx_train_acc_and_loss(
                    model, flat_train_imgs, TRAIN_LABELS)
                dev_acc, dev_loss = dev_acc_and_loss(
                    model, flat_dev_imgs, DEV_LABELS)
                step_metrics = {
                    'step': step,
                    'train_loss': loss.item(),
                    'train_acc': train_acc,
                    'dev_loss': dev_loss,
                    'dev_acc': dev_acc
                }

                print(
                    f"On step {step}:\tTrain loss {train_loss}\t|\tDev acc is {dev_acc}")
                logger.writerow(step_metrics)
        LOGFILE.close()

        # TODO (OPTIONAL) You can remove the date prefix if you don't want to save every model you train
        # i.e. "{DATE_PREFIX}_bestmodel.pt" > "bestmodel.pt"
        model_savepath = os.path.join(
            MODEL_SAVE_DIR, f"{DATE_PREFIX}_bestmodel.pt")

        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)

    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None:
            raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None:
            raise TypeError(
                "for inference, a predictions file must be specified for output.")
        TEST_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_test_images.npy"))

        model = torch.load(WEIGHTS_FILE)

        predictions = []
        for test_case in TEST_IMAGES:
            # TODO implement any normalization schemes you need to apply to your test dataset before inference
            test_case = normalize(test_case)

            x = torch.from_numpy(test_case.astype(np.float32))
            x = x.view(1, -1)
            logits = model(x)
            pred = torch.max(logits, 1)[1]
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")

    else:
        raise Exception("Mode not recognized")
