import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def der_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class Neuron:
    def __init__(self, weight, bias):
        # weight should be an np.array with appropriate number of elements
        self.weight = weight
        self.bias = bias

    def compute(self, x):
        return np.dot(self.weight, x) + self.bias

    def feedforward(self, x):
        return sigmoid(self.compute(x))


X = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
])
Y = np.array([
    [1],
    [0],
    [0],
    [1],
])

no_of_inputs = X.shape[1]
no_of_outputs = Y.shape[1]
no_of_hidden_neurons = 4

HiddenLayer = [Neuron(np.array([np.random.normal()]*no_of_inputs), np.random.normal()) for _ in range(no_of_hidden_neurons)]
OutputLayer = [Neuron(np.array([np.random.normal()]*no_of_hidden_neurons), np.random.normal()) for _ in range(no_of_outputs)]

learn_rate = 0.1
epochs = 1000
for epoch in range(1, epochs+1):
    mse = 0
    for i0 in range(len(X)):
        x = X[i0]
        y = Y[i0]
        h = np.array([hidden_neuron.feedforward(x) for hidden_neuron in HiddenLayer])
        hc = np.array([hidden_neuron.compute(x) for hidden_neuron in HiddenLayer])
        o = np.array([output_neuron.feedforward(h) for output_neuron in OutputLayer])
        oc = np.array([output_neuron.compute(h) for output_neuron in OutputLayer])

        d_L_ok = [2*(o[k] - y[k]) for k in range(no_of_outputs)]
        d_ok_hj = [[OutputLayer[k].weight[j]*der_sigmoid(oc[k]) for j in range(no_of_hidden_neurons)] for k in range(no_of_outputs)]
        d_hj_whji = [[x[i]*der_sigmoid(hc[j]) for i in range(no_of_inputs)] for j in range(no_of_hidden_neurons)]
        d_hj_bhj = [der_sigmoid(hc[j]) for j in range(no_of_hidden_neurons)]
        d_ok_wokj = [[h[j]*der_sigmoid(oc[k]) for j in range(no_of_hidden_neurons)] for k in range(no_of_outputs)]
        d_ok_bok = [der_sigmoid(oc[k]) for k in range(no_of_outputs)]

        d_L_whji = [[sum(d_L_ok[k]*d_ok_hj[k][j]*d_hj_whji[j][i] for k in range(no_of_outputs)) for i in range(no_of_inputs)] for j in range(no_of_hidden_neurons)]
        d_L_bhj = [sum(d_L_ok[k]*d_ok_hj[k][j]*d_hj_bhj[j] for k in range(no_of_outputs)) for j in range(no_of_hidden_neurons)]
        d_L_wokj = [[d_L_ok[k]*d_ok_wokj[k][j] for j in range(no_of_hidden_neurons)] for k in range(no_of_outputs)]
        d_L_bok = [d_L_ok[k]*d_ok_bok[k] for k in range(no_of_outputs)]

        for j in range(no_of_hidden_neurons):
            HiddenLayer[j].bias -= learn_rate * d_L_bhj[j]
            for i in range(no_of_inputs):
                HiddenLayer[j].weight[i] -= learn_rate * d_L_whji[j][i]
        for k in range(no_of_outputs):
            OutputLayer[k].bias -= learn_rate * d_L_bok[k]
            for j in range(no_of_hidden_neurons):
                OutputLayer[k].weight[j] -= learn_rate * d_L_wokj[k][j]

        mse += ((o-y)**2).mean()

    mse /= len(X)
    if epoch%100 == 0: print(f'MSE after epoch {epoch}: {mse}')