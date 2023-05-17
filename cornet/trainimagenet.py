# Training CORnet-S on ImageNet (ILSVRC).
# Code Author: Jay Hill (jdh1g19@soton.ac.uk)
# Model Authors: Kubilius, J. et al. [1, 2]
# Dataset Authors: Princeton University, Stanford University (see end of
#                  file for disclaimer).
# Date: 2023-04-06
# University of Southampton

from cornets import CORnetS, update_model_state_dict

from torchvision import datasets, transforms
import torch

import numpy as np

import time

from PIL import Image

import sys

IMG_TO_TENSOR = transforms.ToTensor()

def image_path_to_tensor(
    path
  ):
  img = Image.open(path)
  tensor = IMG_TO_TENSOR(img)
  if tensor.shape[0] > 3:
    tensor = tensor[:3]
  return tensor.expand(3, *tensor.shape[1:])

def get_imagenet(
    path
  ):
  return datasets.DatasetFolder(
    path,
    image_path_to_tensor,
    ("jpeg",),
    transform = torch.nn.Sequential(
      transforms.RandomCrop((224, 224), pad_if_needed = True),
      transforms.RandomHorizontalFlip(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
  )

def accuracy_from_logits(
    pred_logits, target_batch
  ):
  probabilities = torch.nn.functional.softmax(pred_logits, dim = 1)
  matches = torch.argmax(probabilities, dim = 1) == target_batch
  accuracy = torch.sum(matches) / target_batch.shape[-1]
  return accuracy.item()

def train_cornets(
    cornets,
    epochs,
    loss_function,
    optimizer,
    dataloader,
    epoch_start = 0,
    validation_dataloader = None,
    validation_iters = 1000,
    scheduler = None
  ):
  print("Train start.")
  for epoch in range(epoch_start, epochs):
    print("Epoch:", epoch, "- Start")
    start = time.time()
    avr = 0
    for i, load_batch in enumerate(dataloader):
      image_batch, target_batch = load_batch
      image_batch = image_batch.to("cuda")
      target_batch = target_batch.to("cuda")
      pred_logits = cornets(image_batch)
      loss = loss_function(pred_logits, target_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avr += loss.item() / len(dataloader)
      print(
        "\rIter: %d - Loss:" % i,
        loss.item(),
        "- Avr. Loss:",
        avr * (len(dataloader) / (i + 1)),
        end = ""
      )
      if i % validation_iters == 0 and validation_dataloader is not None:
        print("\nMeasuring accuracy...", end = "")
        cornets.eval()

        with torch.no_grad():
          accuracies = []
          for image_batch_val, target_batch_val in validation_dataloader:
            image_batch_val = image_batch_val.to("cuda", non_blocking = True)
            target_batch_val = target_batch_val.to("cuda", non_blocking = True)
            pred_logits_val = cornets(image_batch_val)
            accuracies.append(
              accuracy_from_logits(pred_logits_val, target_batch_val)
            )
          mean_accuracy = np.mean(accuracies)
          print("\rValidation accuracy:", mean_accuracy)
        cornets.train()
    delta = time.time() - start
    if scheduler is not None:
      scheduler.step(avr)
    print("\nEpoch:", epoch, "- Time: %.2f" % delta, "- Loss:", loss.item())
    torch.save({
      "epoch": epoch,
      "model_state_dict": cornets.state_dict(),
      "optimizer_state_dict": optimizer.state_dict()
    }, "checkpoints/checkpoint%d.tar" % epoch)

def critical_training(model, layers):
  for name, m in model.named_parameters():
    if any(value in name for value in layers):
      #print("YES",name)
      m.requires_grad = True
    else:
      m.requires_grad = False
  return model

if __name__ == "__main__":
  path = sys.argv[1]
  imagenet = get_imagenet(path)
  imagenet_train, imagenet_val = torch.utils.data.random_split(
    imagenet, [1 - 0.001, 0.001],
    #imagenet, [100, len(imagenet) - 100],
    generator = torch.Generator().manual_seed(84)
  )
  dataloader = torch.utils.data.DataLoader(
    imagenet_train,
    shuffle = True,
    batch_size = 128,
    num_workers = 16,
    pin_memory = True
  )
  validation_dataloader = torch.utils.data.DataLoader(
    imagenet_val,
    shuffle = True,
    batch_size = 64,
    num_workers = 4,
    pin_memory = True
  )
  cornets = CORnetS()
  epoch = 0
  optimizer = torch.optim.SGD(
    cornets.parameters(),
    lr = 0.1,
    momentum = 0.9,
    weight_decay = 1e-4
  )
  # Loading stuff.
  checkpoint = None
  if checkpoint is not None:
    checkpoint = torch.load(checkpoint)
    update_model_state_dict(checkpoint["model_state_dict"])
    cornets.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1
  critical_training_layers = None
  #critical_training_layers = ["V2.conv3", "V4.conv3", "IT.conv3", "linear"]
  if critical_training_layers is not None:
    cornets = critical_training(cornets, critical_training_layers)

  cornets.to("cuda")
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = "min",
    factor = 0.1,
    patience = 3,
    threshold = 0.0001,
    verbose = True
  )
  train_cornets(
    cornets,
    50,
    torch.nn.CrossEntropyLoss(),
    optimizer,
    dataloader,
    validation_dataloader = validation_dataloader,
    epoch_start = epoch,
    scheduler = scheduler
  )

# Jay Hill (the "Researcher") has requested permission to use the ImageNet
# database (the "Database") at Princeton University and Stanford University.
# In exchange for such permission, Researcher hereby agrees to the following
# terms and conditions:
#
# 1. Researcher shall use the Database only for non-commercial research and
#    educational purposes.
# 2. Princeton University and Stanford University make no representations or
#    warranties regarding the Database, including but not limited to
#    warranties of non-infringement or fitness for a particular purpose.
# 3. Researcher accepts full responsibility for his or her use of the Database
#    and shall defend and indemnify the ImageNet team, Princeton University,
#    and Stanford University, including their employees, Trustees, officers
#    and agents, against any and all claims arising from Researcher's use of
#    the Database, including but not limited to Researcher's use of any copies
#    of copyrighted images that he or she may create from the Database.
# 4. Researcher may provide research associates and colleagues with access to
#    the Database provided that they first agree to be bound by these terms
#    and conditions.
# 5. Princeton University and Stanford University reserve the right to
#    terminate Researcher's access to the Database at any time.
# 6. If Researcher is employed by a for-profit, commercial entity,
#    Researcher's employer shall also be bound by these terms and conditions,
#    and Researcher hereby represents that he or she is fully authorized to
#    enter into this agreement on behalf of such employer.
# 7. The law of the State of New Jersey shall apply to all disputes under this
#    agreement.