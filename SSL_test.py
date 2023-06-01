# employee_ssl.py
# semi-supervised learning
# predict job from sex, age, city, income
# PyTorch 1.7.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10 

import numpy as np
# import time  # for saving checkpoints
import torch as T
device = T.device('cpu')  # apply to Tensor or Module

# -----------------------------------------------------------

class EmployeeData():
  def __init__(self, src_file, shuffle=True):
    all_xy = np.loadtxt(src_file, usecols=range(0,7),
      delimiter="\t", comments="#", dtype=np.float32)
    tmp_x = all_xy[:,0:6]   # cols [0,6) = [0,5]
    tmp_y = all_xy[:,6]     # 1-D
    self.x_data = T.tensor(tmp_x, dtype=T.float32)  #1-D
    self.y_data = T.tensor(tmp_y, dtype=T.int64)  # ignored 

    self.shuffle = shuffle
    self.rnd = np.random.RandomState(1)
    self.n = len(self.x_data)
    self.indices = np.arange(self.n)
    if self.shuffle == True:
      self.rnd.shuffle(self.indices)
    self.ptr = 0
 
  def get_batch(self, b_size):  # randomly selected
    if self.ptr + b_size > self.n:
      # print("** Resetting ** ")
      if self.shuffle == True:
        self.rnd.shuffle(self.indices)
      self.ptr = 0

    preds = self.x_data[self.indices[self.ptr:self.ptr+b_size]]
    trgts = self.y_data[self.indices[self.ptr:self.ptr+b_size]]
    self.ptr += b_size
    return (preds, trgts)  # as a tuple

# -----------------------------------------------------------

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()  # older syntax
    self.hid1 = T.nn.Linear(6, 10)  # 6-(10-10)-3
    self.hid2 = T.nn.Linear(10, 10)
    self.oupt = T.nn.Linear(10, 3)

    T.nn.init.xavier_uniform_(self.hid1.weight)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.tanh(self.hid1(x))
    z = T.tanh(self.hid2(z))
    z = T.log_softmax(self.oupt(z), dim=1)  # for NLLLoss() 
    return z

# -----------------------------------------------------------

def accuracy(model, data_obj):
  shuffle_state = data_obj.shuffle
  # train_state = model.
  data_obj.shuffle = False
  n = data_obj.n

  n_correct = 0; n_wrong = 0
  for i in range(n):
    (X, y) = data_obj.get_batch(1)
    # print(X)
    # print(y)
    # input()
    with T.no_grad():
      oupt = model(X)  # logits form

    big_idx = T.argmax(oupt)  # 0 or 1 or 2
    if big_idx == y:
      n_correct += 1
    else:
      n_wrong += 1

  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  data_obj.shuffle = shuffle_state
  return acc

# -----------------------------------------------------------

def compute_alpha(batch_num, max_batches):
  if batch_num < int(0.10 * max_batches): return 0.0
  elif batch_num > int(0.90* max_batches): return 1.0
  else: return (batch_num * 1.0) / max_batches  # pct used

# -----------------------------------------------------------

def main():
  # 0. get started
  print("\nBegin Employee semi-supervised predict job ")
  T.manual_seed(1)
  np.random.seed(1)
  
  # 1. create Data objects
  print("\nCreating Employee Data objects ")
  train_file = ".\\Data\\employee_train_labeled.txt"
  train_data_obj = EmployeeData(train_file, True)

  unlabeled_file = ".\\Data\\employee_train_unlabeled.txt"
  unlabeled_data_obj = EmployeeData(unlabeled_file, True)

  test_file = ".\\Data\\employee_test.txt"
  test_data_obj = EmployeeData(test_file, False)

  # 2. create network
  net = Net().to(device)

  # 3. train model using labeled and unlabeled data
  max_batches = 80_000
  bat_log_interval = 10_000
  lrn_rate = 0.01
  bat_size = 4
  loss_func = T.nn.NLLLoss() 
  optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

  print("\nbat_size = %3d " % bat_size)
  print("loss = " + str(loss_func))
  print("optimizer = SGD")
  print("max_batches = %3d " % max_batches)
  print("lrn_rate = %0.3f " % lrn_rate)

  print("\nStarting training")
  net.train()  # or net = net.train()
  acc_loss = 0.0  # monitor accumulated loss

  for bat_num in range(max_batches):
    alpha = compute_alpha(bat_num, max_batches) 

    # labeled loss
    batch = train_data_obj.get_batch(bat_size)
    X = batch[0]  # inputs
    Y = batch[1]     # correct class/label/job
    oupt = net(X)  # as log-probs
    labeled_loss = loss_func(oupt, Y)  # a tensor
 
    # unlabeled loss
    batch = unlabeled_data_obj.get_batch(bat_size)
    X = batch[0]  # inputs
    oupt = net(X)  # as log-probs
    Y = T.argmax(oupt, dim=1)  # pseudo-labels
    unlabeled_loss = loss_func(oupt, Y)  # a tensor

    # combined loss
    loss_val = labeled_loss + (alpha * unlabeled_loss)
    
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    acc_loss += loss_val.item()
    if bat_num % bat_log_interval == 0:
      print("batch num = %7d  |  loss = %0.4f " % (bat_num, loss_val))
      acc_loss = 0.0

      # net.eval()
      # acc = accuracy(net, test_data_obj)
      # print(acc)
      # net.train()
  print("Done ")
 
  # 4. evaluate model accuracy
  print("\nComputing model accuracy ")
  net.eval()

  acc_train = accuracy(net, train_data_obj)  # item-by-item
  print("Accuracy on training data = %0.4f" % acc_train)

  acc_unlabeled = accuracy(net, unlabeled_data_obj) 
  print("Accuracy on unlabeled data = %0.4f" % acc_unlabeled)

  acc_test = accuracy(net, test_data_obj) 
  print("Accuracy on test data = %0.4f" % acc_test)

  # 5. make a prediction
  print("\nPredicting job for male 30  concord  $50,000: ")
  X = np.array([[-1, 0.30,  0,0,1,  0.5000]], dtype=np.float32)
  X = T.tensor(X, dtype=T.float32).to(device) 

  with T.no_grad():
    log_probs = net(X)
  probs = T.exp(log_probs)  # tensor
  probs = probs.numpy()     # array
  np.set_printoptions(precision=6, suppress=True)
  print(probs)

  # 6. save model (state_dict approach)
  print("\nSaving trained model state")
  fn = ".\\Models\\employee_ssl_model.pt"
  T.save(net.state_dict(), fn)

  # saved_model = Net()
  # saved_model.load_state_dict(T.load(fn))
  # use saved_model to make prediction(s)

  print("\nEnd Employee predict job demo")

if __name__ == "__main__":
  main()