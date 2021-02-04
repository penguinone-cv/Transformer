import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from parameter_loader import *
from model import TransformerClassification
from logger import Logger
from optimizer import NoamOpt
from dataloader import *

# For data loading.
from torchtext import data, datasets



class Trainer:
    def __init__(self, setting_csv_path, index):
        parameters_dict = read_parameters(setting_csv_path, index)
        self.model_name = parameters_dict["model_name"]
        log_dir_name = self.model_name + "_epochs" + parameters_dict["epochs"] \
                            + "_batch_size" + parameters_dict["batch_size"] \
                            + "_lr" + parameters_dict["learning_rate"]                         #ログを保存するフォルダ名
        self.log_path = os.path.join(parameters_dict["base_log_path"], log_dir_name)      #ログの保存先
        self.data_path = parameters_dict["data_path"]
        self.batch_size = int(parameters_dict["batch_size"])
        self.epochs = int(parameters_dict["epochs"])
        self.learning_rate = float(parameters_dict["learning_rate"])
        self.layers_num = int(parameters_dict["layers_num"])
        self.d_model = int(parameters_dict["d_model"])
        self.d_ff = int(parameters_dict["d_ff"])
        self.heads = int(parameters_dict["h"])
        self.dropout_rate = float(parameters_dict["dropout_rate"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                  #GPUが利用可能であればGPUを利用

        self.dataloader = DataLoader(dataset_path=self.data_path, batch_size=self.batch_size)
        self.dataloaders_dict = {"train": self.dataloader.train_dl, "val": self.dataloader.val_dl}

        self.model = TransformerClassification(text_embedding_vectors=self.dataloader.TEXT.vocab.vectors, heads=self.heads, layers_num=self.layers_num,
                                                dropout_rate=self.dropout_rate, d_model=300, d_ff=self.d_ff, max_seq_len=256, output_dim=2)
        self.logger = Logger(self.log_path)                                                         #ログ書き込みを行うLoggerクラスの宣言

        #self.src_vocab, self.target_vocab =

        #self.model = make_model()

    def train(self):
        self.model.train()
        for i in range(self.layers_num):
            self.model.encoders[i].apply(self.weights_init)
        #self.model.net3_1.apply(self.weights_init)
        #self.model.net3_2.apply(self.weights_init)
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        learning_rate = 2e-5
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        torch.backends.cudnn.benchmark = True

        with tqdm(range(self.epochs)) as progress_bar:
            for epoch in enumerate(progress_bar):
                i = epoch[0]
                progress_bar.set_description("[Epoch %d]" % (i+1))
                epoch_loss = 0.
                epoch_corrects = 0
                val_loss_result = 0.0
                val_acc = 0.0

                self.model.train()
                j = 1
                for batch in self.dataloaders_dict["train"]:
                    progress_bar.set_description("[Epoch %d (Iteration %d)]" % ((i+1), j))
                    j = j + 1
                    inputs = batch.Text[0].to(self.device)
                    labels = batch.Label.to(self.device)

                    input_pad = 1
                    input_mask = (inputs != input_pad)
                    input_mask = input_mask.to(self.device)

                    outputs, _, _ = self.model(inputs, input_mask)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

                else:
                    with torch.no_grad():
                        self.model.eval()
                        progress_bar.set_description("[Epoch %d (Validation)]" % (i+1))
                        for val_batch in self.dataloaders_dict["val"]:
                            inputs = val_batch.Text[0].to(self.device)
                            labels = val_batch.Label.to(self.device)

                            input_pad = 1
                            input_mask = (inputs != input_pad)
                            input_mask = input_mask.to(self.device)

                            outputs, _, _ = self.model(inputs, input_mask)
                            loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)
                            val_loss_result += loss.item()
                            val_acc += torch.sum(preds == labels.data)

                    epoch_loss = epoch_loss / len(self.dataloaders_dict["train"].dataset)
                    epoch_acc = epoch_corrects.float() / len(self.dataloaders_dict["train"].dataset)
                    val_epoch_loss = val_loss_result / len(self.dataloaders_dict["val"].dataset)
                    val_epoch_acc = val_acc.float() / len(self.dataloaders_dict["val"].dataset)
                    self.logger.collect_history(loss=epoch_loss, accuracy=epoch_acc, val_loss=val_epoch_loss, val_accuracy=val_epoch_acc)
                    self.logger.writer.add_scalars("losses", {"train":epoch_loss,"validation":val_epoch_loss}, (i+1))
                    self.logger.writer.add_scalars("accuracies", {"train":epoch_acc, "validation":val_epoch_acc}, (i+1))

                progress_bar.set_postfix({"loss":epoch_loss, "accuracy": epoch_acc.item(), "val_loss":val_epoch_loss, "val_accuracy": val_epoch_acc.item()})

        torch.save(self.model.state_dict(), os.path.join(self.log_path,self.model_name))
        self.logger.draw_graph()
        self.logger.writer.flush()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
