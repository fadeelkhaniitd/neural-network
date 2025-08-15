## On `NN_from_scratch.py`

This is a code implementing a neural network (with one hidden layer) from scratch using NumPy.

### The math behind it

For each row of data, we have the mean squared loss $$L = \frac{1}{n_{out}} \sum_{k=1}^{n_{out}}(y_k - o_k)^2$$

where $$o_k = f(w_{o_k} \cdot h + b_{o_{k}})$$
$$h = [h_1, h_2, \dots]$$
$$h_j = f(w_{h_j} \cdot x + b_{h_{j}})$$
$$x = [x_1, x_2, \dots]$$

$k$ index is for output neurons, $j$ index is for hidden neurons, and $i$ index is for inputs. $f$ is the activation function.

We have to compute the partial derivatives.

$$\frac{\partial L}{\partial w_{h_{ji}}} = \sum_{k=1}^{n_{out}} \frac{\partial L}{\partial o_k} \cdot \frac{\partial o_k}{\partial h_{j}} \cdot \frac{\partial h_{j}}{\partial w_{h_{ji}}} $$
$$\frac{\partial L}{\partial o_k} = 2(o_k - y_k) $$
$$\frac{\partial o_k}{\partial h_j} = w_{o_{kj}} f'(w_{o_k} \cdot h + b_{o_k}) $$
$$\frac{\partial h_j}{\partial w_{h_{ji}}} = x_i f'(w_{h_j} \cdot x + b_{h_j})$$
$$ $$
$$\frac{\partial L}{\partial b_{h_{j}}} = \sum_{k=1}^{n_{out}} \frac{\partial L}{\partial o_k} \cdot \frac{\partial o_k}{\partial h_{j}} \cdot \frac{\partial h_{j}}{\partial b_{h_{j}}} $$
$$\frac{\partial L}{\partial o_k} = 2(o_k - y_k) $$
$$\frac{\partial o_k}{\partial h_j} = w_{o_{kj}} f'(w_{o_k} \cdot h + b_{o_k}) $$
$$\frac{\partial h_j}{\partial b_{h_{j}}} = f'(w_{h_j} \cdot x + b_{h_j})$$
$$ $$
$$\frac{\partial L}{\partial w_{o_{kj}}} = \sum_{k=1}^{n_{out}} \frac{\partial L}{\partial o_k} \cdot \frac{\partial o_k}{\partial w_{o_{kj}}} $$
$$\frac{\partial L}{\partial o_k} = 2(o_k - y_k) $$
$$\frac{\partial o_k}{\partial w_{o_{kj}}} = h_j f'(w_{o_k} \cdot h + b_{o_k})$$
$$ $$
$$\frac{\partial L}{\partial b_{o_{k}}} = \sum_{k=1}^{n_{out}} \frac{\partial L}{\partial o_k} \cdot \frac{\partial o_k}{\partial b_{o_{k}}} $$
$$\frac{\partial L}{\partial o_k} = 2(o_k - y_k) $$
$$\frac{\partial o_k}{\partial b_{o_{k}}} = f'(w_{o_k} \cdot h + b_{o_k})$$

In the code, $\frac{\partial y}{\partial x}$ is denoted as `d_y_x`.

---

## On `Fashion_MNIST_NN_from_scratch.ipynb`

We cannot use the naive code implementation in `NN_from_scratch.py` for the above math, because the number of rows of data and the number of input are quite large.

So, I wrote another code which uses fast NumPy computations, something which the previous code didn't.
The code still does not use any other library like pytorch or tensorflow.

After training till my loss function didnt change by much, I got a classification accuracy of 85.38% on the test dataset (loss of 0.022).
