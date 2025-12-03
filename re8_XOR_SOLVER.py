import torch
import matplotlib.pyplot as plt


def l_relu(x):
    return torch.where(x>0, x, x*0.01)

class MLinear():
    def __init__(self, prev_neurons, current_neurons):
        self.weights = torch.empty(prev_neurons, current_neurons) 
        torch.nn.init.kaiming_normal_(self.weights, a=0.01, nonlinearity='leaky_relu')
        self.bias = torch.zeros((current_neurons,))

    def forward(self, input):
        return ( input @ self.weights ) + self.bias
    
class MNetwork():
    def __init__(self):
        self.layers = [
            MLinear(2,2), 
            MLinear(2,1)
        ]
        self.inter = []
    
    def forward(self, input):
        self.inter = [{'a':input}]
        for l in self.layers:
            input = l.forward(input)
            self.inter.append({'wsum':input})
            input = l_relu(input)
            self.inter[-1]['a'] = input
        return input
    
class MSELoss():
    def __init__(self):
        pass

    def get_loss(self, logits, targets):
        loss = (logits - targets) ** 2
        return loss.mean()
    
    def back(self, model, targets):
        
        b_grads=[]
        w_grads=[]

        c__a_p = 2 * ( model.inter[-1]['a'] - targets) / batch_size
        moving_gradient = c__a_p

        for idx in range( len(model.inter)-1, 0, -1 ):

            a__wsum_p = torch.where( model.inter[idx]['wsum']>0, 1.0, 0.01 )
            moving_gradient *= a__wsum_p

            b_grads.insert(0, moving_gradient.sum(dim=0) ) 
            
            wsum__w_p = model.inter[idx-1]['a']
            w_grads.insert(0,    wsum__w_p.T @ moving_gradient )

            wsum__a_p = model.layers[idx-1].weights
            moving_gradient = moving_gradient @ wsum__a_p.T
            
        return w_grads, b_grads

class AdamW():
    def __init__(self, model, beta1=0.9, beta2=0.99, eps=1e-8,weight_decay=0.01):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0 # timestep

        '''
        beta1 (0.9): Controls how much to trust the current gradient vs. past momentum (like "how quickly to forget old steps").

        beta2 (0.99): Controls how much to scale step sizes based on gradient history (like "how aggressively to adapt to steepness").

        eps (1e-8): A tiny number to avoid division-by-zero.

        weight_decay (0.01): How strongly to penalize large weights (to prevent overfitting).

        t: A time counter (starts at 0).

        grad_mean_* and variance_*: Initialize buffers to track momentum and gradient magnitudes for weights/biases.
        '''

        # INITIALIZE TORCH.ZERO_LIKE
        self.grad_mean_w = [torch.zeros_like(l.weights) for l in model.layers] 
        self.variance_w = [torch.zeros_like(l.weights) for l in model.layers ]

        self.grad_mean_b = [torch.zeros_like(l.bias) for l in model.layers]
        self.variance_b = [torch.zeros_like(l.bias)for l in model.layers]

    def step(self, model, w_grads, b_grads):
        self.t +=1
        
        for i, layer in enumerate(model.layers):
            # CALCULATE MOMENTUM/variance
            '''
            Momentum (grad_mean): A smoothed version of past gradients.
            Like a ball rolling downhill, it builds speed in consistent directions:
            '''
            self.grad_mean_w[i] = (self.beta1 * self.grad_mean_w[i]) + ((1-self.beta1) * w_grads[i])
            self.variance_w[i] = (self.beta2 * self.variance_w[i]) + ( (1-self.beta2) * (w_grads[i]**2)  )

            # BIAS CORRECTION
            '''
            Early in training, momentum/variance estimates are unreliable (because t is small). We adjust for this:
            '''
            grad_mean_hat = self.grad_mean_w[i] / (1 - self.beta1 ** self.t)
            variance_hat = self.variance_w[i] / (1 - self.beta2 ** self.t)

            # UPDATE
            '''
            Adaptive step size: Divide by sqrt(variance_hat) to take smaller steps where gradients vary wildly.
            Weight decay: Directly shrink weights (unlike Adam, where it’s mixed with gradients).
            '''
            layer.weights -= lr * ( grad_mean_hat / (  torch.sqrt(variance_hat) + self.eps) + self.weight_decay * layer.weights )


            # SAME FOR BIAS
            self.grad_mean_b[i] = (self.beta1 * self.grad_mean_b[i]) + ( (1-self.beta1) * b_grads[i] ) 
            self.variance_b[i] = (self.beta2 * self.variance_b[i]) + (1 - self.beta2) * (b_grads[i]**2)

            # CORRECTION
            grad_mean_hat_b = self.grad_mean_b[i] / (1 - self.beta1 ** self.t)
            variance_hat_b = self.variance_b[i] / (1 - self.beta2 ** self.t)

            # UPDATE
            layer.bias -= lr * ( grad_mean_hat_b / ( torch.sqrt(variance_hat_b) + self.eps) + self.weight_decay * layer.bias)


def generate_xor_samples(n_samples):
    q = torch.bernoulli(  torch.full((n_samples,2), 0.5 ) )
    a = (q[:,0] != q[:,1]).float()    
    return q, a 

def infer():

    while True:
        i1 = float(input('Enter 1 OR 0 : '))
        i2 = float(input('Enter 1 OR 0 : '))
        ans = float(i1!=i2)

        pred = model.forward( torch.tensor([[i1, i2]]) )
        print(f'Correct answer:{ans} \n Model prediction [O__O] ---> {pred}')

        if input('break?(y/n)') == 'y':
            break



lr = 0.01
model = MNetwork()
loss_obj = MSELoss()
batch_size = 8
x=[]
y=[]
# ax,fix = plt.subplots()
loss_array = []
optimizer = AdamW(model=model)

for i in range(50000):
    q , a = generate_xor_samples(n_samples=batch_size)
    pred = model.forward( q )
    loss = loss_obj.get_loss(logits = pred, targets = a.view(-1, pred.shape[-1] ))
    w_grads, b_grads = loss_obj.back( model = model, targets = a.view(-1, pred.shape[-1]))
    optimizer.step(model=model, w_grads=w_grads, b_grads=b_grads)
    x.append(i)
    y.append(loss.item())
    loss_array.append(loss)
    loss_array = loss_array[-500:]
    avg_loss = sum(loss_array) / len(loss_array) if loss_array else 0
    max_loss = max(loss_array)
    if avg_loss < 0.01 and len(loss_array)>100 and max_loss < 0.05:
        break
    
    if i%100==0:
        print(f'step:{i}')

plt.plot(x,y)
plt.show()
infer()


if False:
    ag_model = torch.nn.Sequential(
        torch.nn.Linear(2,2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(2,1),
        torch.nn.LeakyReLU()
    )
    with torch.no_grad():
        ag_model[0].weight.copy_( model.layers[0].weights.T )
        ag_model[0].bias.copy_(model.layers[0].bias)

        ag_model[2].weight.copy_(model.layers[1].weights.T)
        ag_model[2].bias.copy_(model.layers[1].bias)

    ag_pred = ag_model(q)
    ag_model.zero_grad()
    ag_loss_fn = torch.nn.MSELoss(reduction='mean')
    ag_loss = ag_loss_fn(ag_pred, a.view(-1, ag_pred.shape[-1]))
    ag_loss.backward()
    ag_w_grad = [
        ag_model[0].weight.grad.T,
        ag_model[2].weight.grad.T
    ]
    ag_b_grad = [
        ag_model[0].bias.grad,
        ag_model[2].bias.grad
    ]
    print(f'AUTOGRAD WEIGHT GRADS:{ag_w_grad} \n AUTOGRAD BIAS GRADS:{ag_b_grad}')
    #print(f'ag_model pred:{ag_pred}')










