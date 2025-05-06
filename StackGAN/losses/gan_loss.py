import torch
import torch.nn as nn

def generator_loss(fake_preds):
    criterion = nn.BCEWithLogitsLoss()
    real_labels = torch.ones_like(fake_preds)
    return criterion(fake_preds, real_labels)

def discriminator_loss(real_preds, fake_preds):
    criterion = nn.BCEWithLogitsLoss()
    real_labels = torch.ones_like(real_preds)
    fake_labels = torch.zeros_like(fake_preds)
    real_loss = criterion(real_preds, real_labels)
    fake_loss = criterion(fake_preds, fake_labels)
    return real_loss + fake_loss
