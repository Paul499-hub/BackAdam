import torch



def relu(data):
    return torch.clamp(data, min=0.0)


class MLinear():
    def __init__(self, prev_neurons, current_neurons):
        self.weights = torch.randn(prev_neurons, current_neurons)
        self.bias = torch.ones((current_neurons,))

    def forward(self, input):
        return ( input @ self.weights ) + self.bias
    
class MNetwork():
    def __init__(self):
        self.layers = [
            MLinear(2,2),
            MLinear(2,2),
            MLinear(2,2),
        ]

    def forward(self, input):
        out = input
        for l in self.layers:
            out = l.forward(out)
            print(f'w_sum (not-activated) layer:{out}')
            out = relu(out)
            print(f'a (activated) layer:{out} \n\n')
        return out
    
model = MNetwork()
# WEIGHTS ARE COLUMN WISE (FIRST NEURON -> FIRST COLUMN... CONNECTIONS REPESENT CONNECTION TO PREVIOUS LAYER)
model.layers[0].weights = torch.tensor([[0.05, 0.06],
                                        [0.1,  0.11]]) 
model.layers[1].weights  = torch.tensor([[0.15, 0.16],
                                         [0.2, 0.21]]) 
model.layers[2].weights  = torch.tensor([[0.25, 0.26],
                                         [0.3, 0.31]]) 

out = model.forward(torch.tensor([2.0, 5.0]))

print(out)




