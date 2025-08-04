Optimizer  
Comparison of similarities and differences between optimizers supported by both PyTorch and MindSpore can be found in the API mapping table.

Differences in optimizer execution and usage  
When executing an optimizer step in PyTorch, you generally need to manually call the zero_grad() method to set historical gradients to zero, then use loss.backward() to compute the gradients for the current training step, and finally call the optimizer's step() method to update the network weights.

When using an optimizer in MindSpore, you only need to compute the gradients directly and then use optimizer(grads) to update the network weights.

If you need to dynamically adjust the learning rate during training, PyTorch provides the LRScheduler class for learning rate management. When using a dynamic learning rate, pass the optimizer instance into a subclass of LRScheduler, and loop through scheduler.step() to modify the learning rate and synchronize the changes to the optimizer.

MindSpore provides two methods for dynamic learning rate adjustment: Cell and list. When using, pass the dynamic learning rate object directly into the optimizer, and the learning rate update is automatically performed within the optimizer. For details, refer to dynamic learning rate.

PyTorch
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

optimizer.zero_grad()
output = model(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
scheduler.step()
```
MindSpore

```python
import mindspore
from mindspore import nn

lr = nn.exponential_decay_lr(0.01, decay_rate, total_step, step_per_epoch, decay_epoch)

optimizer = nn.SGD(model.trainable_params(), learning_rate=lr, momentum=0.9)
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
(loss, _), grads = grad_fn(data, label)
# Learning rate update is done automatically in the optimizer
optimizer(grads)
```

Automatic Differentiation  
Both MindSpore and PyTorch provide automatic differentiation, allowing you to perform automatic backpropagation and gradient updates with simple interface calls after defining the forward network. Note that MindSpore and PyTorch construct the backward graph differently, which leads to differences in API design.

PyTorch Automatic Differentiation  
# torch.autograd:
# backward accumulates gradients, so optimizer needs to be cleared after updating
```python
import torch.nn as nn
import torch.optim as optim

# Instantiate model and optimizer

model = PT_Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define loss function: Mean Squared Error (MSE)

loss_fn = nn.MSELoss()

# Forward pass: compute model output
y_pred = model(x)

# Compute loss: calculate loss between predictions and true labels
loss = loss_fn(y_pred, y_true)

# Backward pass: compute gradients
loss.backward()
# Optimizer update
optimizer.step()
```
MindSpore Automatic Differentiation  
# ms.grad:
# Use grad interface, input forward graph, output backward graph
```python
import mindspore as ms
from mindspore import nn

# Instantiate model and optimizer
model = MS_Model()
optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01)

# Define loss function: Mean Squared Error (MSE)
loss_fn = nn.MSELoss()

def forward_fn(x, y_true):
    # Forward pass: compute model output
    y_pred = model(x)
    # Compute loss: calculate loss between predictions and true labels
    loss = loss_fn(y_pred, y_true)
    return loss, y_pred

# Compute loss and gradients
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
(loss, _), grads = grad_fn(x, y_true)
# Optimizer update
optimizer(grads)
```

Model Training and Inference  
Below is an example of a Trainer in MindSpore, including training and inference during training. The training part mainly involves combining dataset, model, optimizer, etc. for training; the inference part mainly involves obtaining evaluation metrics and saving the best model parameters.
```python
import mindspore as ms
from mindspore import nn
from mindspore.amp import StaticLossScaler, all_finite
from mindspore.communication import init, get_group_size

class Trainer:
    """A training example with two losses"""
    def __init__(self, net, loss1, loss2, optimizer, train_dataset, loss_scale=1.0, eval_dataset=None, metric=None):
        self.net = net
        self.loss1 = loss1
        self.loss2 = loss2
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.train_data_size = self.train_dataset.get_dataset_size()    # Get number of batches in training set
        self.weights = self.opt.parameters
        # Note: The first argument of value_and_grad should be the graph to compute gradients for, usually including network and loss. It can be a function or a Cell.
        self.value_and_grad = ms.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)

        # For distributed scenarios
        self.grad_reducer = self.get_grad_reducer()
        self.loss_scale = StaticLossScaler(loss_scale)
        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.metric = metric
            self.best_acc = 0

    def get_grad_reducer(self):
        grad_reducer = nn.Identity()
        # Check if distributed scenario; for distributed settings, refer to general environment setup above
        group_size = get_group_size()
        reducer_flag = (group_size != 1)
        if reducer_flag:
            grad_reducer = nn.DistributedGradReducer(self.weights)
        return grad_reducer

    def forward_fn(self, inputs, labels):
        """Forward network construction; note the first output must be the one to compute gradients for"""
        logits = self.net(inputs)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        loss = self.loss_scale.scale(loss)
        return loss, loss1, loss2

    @ms.jit    # JIT acceleration; must meet graph mode requirements, otherwise will throw error
    def train_single(self, inputs, labels):
        (loss, loss1, loss2), grads = self.value_and_grad(inputs, labels)
        loss = self.loss_scale.unscale(loss)
        grads = self.loss_scale.unscale(grads)
        grads = self.grad_reducer(grads)
        self.opt(grads)
        return loss, loss1, loss2

    def train(self, epochs):
        train_dataset = self.train_dataset.create_dict_iterator(num_epochs=epochs)
        self.net.set_train(True)
        for epoch in range(epochs):
            # Train one epoch
            for batch, data in enumerate(train_dataset):
                loss, loss1, loss2 = self.train_single(data["image"], data["label"])
                if batch % 100 == 0:
                    print(f"step: [{batch} /{self.train_data_size}] "
                          f"loss: {loss}, loss1: {loss1}, loss2: {loss2}", flush=True)
            # Save model and optimizer weights for current epoch
            ms.save_checkpoint(self.net, f"epoch_{epoch}.ckpt")
            ms.save_checkpoint(self.opt, f"opt_{epoch}.ckpt")
            # Inference and save the best checkpoint
            if self.run_eval:
                eval_dataset = self.eval_dataset.create_dict_iterator(num_epochs=1)
                self.net.set_train(False)
                self.eval(eval_dataset, epoch)
                self.net.set_train(True)

    def eval(self, eval_dataset, epoch):
        self.metric.clear()
        for batch, data in enumerate(eval_dataset):
            output = self.net(data["image"])
            self.metric.update(output, data["label"])
        accuracy = self.metric.eval()
        print(f"epoch {epoch}, accuracy: {accuracy}", flush=True)
        if accuracy >= self.best_acc:
            # Save the best checkpoint
            self.best_acc = accuracy
            ms.save_checkpoint(self.net, "best.ckpt")
            print(f"Update best acc: