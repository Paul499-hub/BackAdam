import torch
import matplotlib.pyplot as plt 


# torch.manual_seed(40)
def l_relu(x):
    return torch.where(x>0, x, x*0.01)

class MLinear():
    def __init__(self, p_neurons, c_neurons):
        self.weights = torch.empty(p_neurons, c_neurons)
        torch.nn.init.kaiming_normal_(self.weights, a=0.01, nonlinearity='leaky_relu')
        self.bias = torch.zeros((c_neurons,))
    def forward(self, x):
        return (x @ self.weights) + self.bias
    
class MNetwork():
    def __init__(self):
        self.layers = [
            MLinear(2,2),
            MLinear(2,1)
        ]
        self.inter = None

    def forward(self, x):
        self.inter = [{'a':x}]
        for l in self.layers:
            x = l.forward(x)
            self.inter.append({'wsum':x})

            x = l_relu(x)
            self.inter[-1]['a'] = x
        return x

class MSELoss():
    def __init__(self):
        pass

    def get_loss(self, logits, targets):
        loss = (logits - targets) ** 2 
        return loss.mean()
     
    def back(self, model, targets):

        w_grads = []
        b_grads = []
        moving_gradient = None
        # COST WITH RESPECT TO A partial derivative 
        c__a_p = 2 * ( model.inter[-1]['a'] - targets) / batch_size
        moving_gradient = c__a_p
        for idx in range(  len(model.inter)-1, 0, -1):
            # ACTIVATIONS WITH RESPECT TO wsum partial derivative 
            a__wsum_p =  torch.where( model.inter[idx]['wsum']> 0, 1.0, 0.01 )
            # print(f' a__wsum_p shape:{a__wsum_p.shape}')
            # print(f' moving gradient shape:{moving_gradient.shape}')
            moving_gradient *= a__wsum_p 
            
            # Moving gradient shape: [2,1]
            b_grads.insert(0,moving_gradient.sum(dim=0))
            
            # wsum WITH RESPECT TO w PARTIAL DERIVATIVE 
            wsum__w_p = model.inter[idx-1]['a']
            # print(f' wsum__w_p shape:{wsum__w_p.shape}')# OUTPUT:  wsum__w_p shape:torch.Size([2, 2])
            # print(f' moving gradient shape: {moving_gradient.shape}') # OUTPUT:  moving gradient shape: torch.Size([2, 1])
            w_grads.insert(0, wsum__w_p.T @ moving_gradient )

            # I NEED CURRENT LAYER's WEIGHTS HERE, but model.layers(2) != model.inter(3 cause inputs are considered last layer), THEREFORE we -1 THE INDEX
            wsum__a_p = model.layers[idx-1].weights
            #print(f' model layer custom idx weights: {model.layers[idx-1].weights} \n SHOULD MATCH: { model.inter[idx]["w"]} ') # OUTPUT: THEY DO MATCH :) 
            #print(f'moving gradient shape:{moving_gradient.shape}  wsum__a_p shape:{wsum__a_p.shape}')# OUTPUT moving_gradient: [2,1] , wsum__a_p:[2,1]
            moving_gradient = moving_gradient @ wsum__a_p.T
        return w_grads, b_grads

class SGD():
    def __init__(self):
        pass

    def step(self, model, w_grads, b_grads):

        for idx in range(len(model.layers)):
            model.layers[idx].weights -= w_grads[idx] * lr
            model.layers[idx].bias -= b_grads[idx] * lr 

# GTP
if False:
    class AdamW():
        def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
            self.lr = lr
            self.beta1 = betas[0]
            self.beta2 = betas[1]
            self.eps = eps
            self.weight_decay = weight_decay
            
            # Initialize moment estimates for all params (weights and biases)
            self.m_w = [torch.zeros_like(layer.weights) for layer in model.layers]
            self.v_w = [torch.zeros_like(layer.weights) for layer in model.layers]
            self.m_b = [torch.zeros_like(layer.bias) for layer in model.layers]
            self.v_b = [torch.zeros_like(layer.bias) for layer in model.layers]
            
            self.t = 0  # timestep counter
        
        def step(self, model, w_grads, b_grads):
            self.t += 1
            for i, layer in enumerate(model.layers):
                # Update moments for weights
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * w_grads[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (w_grads[i] ** 2)
                
                # Bias correction
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                
                # Update weights with AdamW rule
                layer.weights = layer.weights - self.lr * (m_w_hat / (torch.sqrt(v_w_hat) + self.eps) + self.weight_decay * layer.weights)
                
                # Same for biases
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * b_grads[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (b_grads[i] ** 2)
                
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                
                layer.bias = layer.bias - self.lr * (m_b_hat / (torch.sqrt(v_b_hat) + self.eps) + self.weight_decay * layer.bias)

class AdamW():
    def __init__(self, model, beta1=0.9, beta2=0.999 , eps=1e-8, weight_decay=0.01 ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # first moment m(mean of gradients) v(avg of gradients squared)
        self.m_w = [ torch.zeros_like(l.weights)  for l in model.layers ]
        self.m_b = [ torch.zeros_like(l.bias) for l in model.layers ]

        self.w_v = [ torch.zeros_like(l.weights)  for l in model.layers ]
        self.b_v = [ torch.zeros_like(l.bias) for l in model.layers ]

        self.t = 0 # timestep counter

    def step(self, model, w_grads, b_grads):
        self.t +=1

        for i, layer in enumerate(model.layers):
            # β₁: how “stubborn” you are about changing direction.
            # β₂: how much you trust the current estimate of scale (adaptivity).
            
            # WEIGHTS
            # Calculate momentum's 
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * w_grads[i]
            self.w_v[i] = self.beta2 * self.w_v[i] + (1 - self.beta2) * (w_grads[i] ** 2)

            #Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            w_v_hat = self.w_v[i] / (1 - self.beta2 ** self.t)

            # Update
            layer.weights -= lr * ( m_w_hat / ( torch.sqrt(w_v_hat) + self.eps ) + self.weight_decay * layer.weights)

            #BIAS
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * b_grads[i]
            self.b_v[i] = self.beta2 * self.b_v[i] + (1 - self.beta2) * (b_grads[i] ** 2 )

            # Bias correction
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            b_v_hat = self.b_v[i] / (1 - self.beta2 ** self.t)

            #Update
            layer.bias -= lr * (m_b_hat / ( torch.sqrt(b_v_hat) + self.eps ) + self.weight_decay * layer.bias )

def generate_xor_samples(batch_size):
    pred = 0.5
    q = torch.bernoulli( torch.full( (batch_size,2), pred  ))
    a = (q[:,0]!=q[:,1]).float()
    if False: # PRINT q a , shapes  
        print(f'q:{q} \n a:{a}')
        print(f'q shape: {q.shape}  a shape: {a.shape}')
    return q, a

def infer():
    while True:
        i1 = float(input('Enter 1 OR 0 : '))
        i2 = float(input('Enter 1 OR 0 : '))

        inputs = torch.tensor( [[i1, i2],[0, 0]])
        pred = model.forward(inputs)
        print(f' Model output (LOOK ONLY FIRST NUMBER):{pred}')
        print(f' Expected output:{ float(i1!=i2)}')

        if input('break?(y/n)')=='y':
            break

model = MNetwork()
loss_obj = MSELoss()
#optimizer = SGD()
optimizer = AdamW(model=model)
batch_size = 1
lr = 0.01
plt_x,plt_y = [],[]
fig,ax = plt.subplots()

for i in range(32000):

    q, a = generate_xor_samples(batch_size = batch_size)
    pred = model.forward(q)
    loss = loss_obj.get_loss( logits = pred, targets = a.view(-1, pred.shape[-1]))
    w_grads,b_grads = loss_obj.back( model=model, targets= a.view(-1, pred.shape[-1] ))
    optimizer.step( model=model, w_grads=w_grads, b_grads=b_grads)

    plt_x.append(i)
    plt_y.append(loss.item())

    if False: # PRINT PREDICTION, a q loss 
        print(f' q:{q}  a:{a}')
        print(f' prediction [O__O]:{pred}')
        print(f'loss:{loss}')
plt.plot(plt_x,plt_y)
plt.show()
# infer()





if False: # AUTOGRAD MODEL COPY TO CHECK IF GRADIENTS MATCH
    a_g_model = torch.nn.Sequential(
        torch.nn.Linear(2,2),
        torch.nn.LeakyReLU(0.01),
        torch.nn.Linear(2,1),
        torch.nn.LeakyReLU(0.01)
    )
    with torch.no_grad():
        
        a_g_model[0].weight.copy_(model.layers[0].weights.T)
        a_g_model[0].bias.copy_(model.layers[0].bias)

        a_g_model[2].weight.copy_(model.layers[1].weights.T)
        a_g_model[2].bias.copy_(model.layers[1].bias)

        if False: # PRINT COPYING OPERATION EXTENSIVELY
            print(f'W LAYER 1 ---- \nautograd model weights layer: {a_g_model[0].weight.T}') # autograd model weights layer: torch.Size([2, 2])
            print(f'CUSTOM model weights layer: {model.layers[0].weights}') # CUSTOM model weights layer: torch.Size([2, 2])

            print(f' W LAYER 2 ---- \nautograd model weights layer: {a_g_model[2].weight.T}') # autograd model weights layer: torch.Size([1, 2])
            print(f'CUSTOM model weights layer: {model.layers[1].weights}')# CUSTOM model weights layer: torch.Size([2, 1])

            print(f'\n BIAS LAYER 1--------- autograd model bias layer: {a_g_model[0].bias}') # torch.Size([2])
            print(f'CUSTOM model weights layer: {model.layers[0].bias}') # torch.Size([2])

            print(f'\n BIAS LAYER 2--------- autograd model bias layer: {a_g_model[2].bias}') # torch.Size([1])
            print(f'CUSTOM model weights layer: {model.layers[1].bias}') # torch.Size([1])
    autograd_pred = a_g_model(q)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    autograd_loss = loss_fn(autograd_pred, a.view(-1,1))
    a_g_model.zero_grad()
    autograd_loss.backward()
    autograd_w_grads = [
        a_g_model[0].weight.grad.T,
        a_g_model[2].weight.grad.T
    ]

    autograd_b_grads = [
        a_g_model[0].bias.grad,
        a_g_model[2].bias.grad
    ]
    autograd_optimizer = torch.optim.SGD(a_g_model.parameters(), lr=lr) 

    with torch.no_grad():
        autograd_pred = a_g_model(q)
        print(f'autograd model pred after SGD:{autograd_pred}')


if False: # TRAINING LOOP AUTOGRAD PYTORCH
    batch_size = 1
    lr = 0.01
    plt_x,plt_y = [],[]
    fig,ax = plt.subplots()
    a_g_model = torch.nn.Sequential(
        torch.nn.Linear(2,2),
        torch.nn.LeakyReLU(0.01),
        torch.nn.Linear(2,1),
        torch.nn.LeakyReLU(0.01)
    )
    autograd_optimizer = torch.optim.SGD(a_g_model.parameters(), lr=lr)
    # autograd_optimizer = torch.optim.AdamW(a_g_model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    for i in range(10000):
        q,a = generate_xor_samples(batch_size=batch_size)
        '''
        q:tensor([[0., 1.],
            [1., 1.]])  a:tensor([1., 0.])
        prediction [O__O]:tensor([[-0.0041],
            [-0.0008]], grad_fn=<LeakyReluBackward0>)
            '''
        autograd_pred = a_g_model(q)
        autograd_loss = loss_fn(autograd_pred, a.view(-1,1))
        a_g_model.zero_grad()
        autograd_loss.backward()
        autograd_optimizer.step()

        if False: # PRINT PREDICTION, a q loss 
            print(f' q:{q}  a:{a}')
            print(f' prediction [O__O]:{autograd_pred}')
            print(f'loss:{autograd_loss}')

        plt_x.append(i)
        plt_y.append(autograd_loss.item())

    plt.plot(plt_x, plt_y)
    plt.show()

