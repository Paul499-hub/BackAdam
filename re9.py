import torch
import matplotlib.pyplot as plt 


def l_relu(input):
    return torch.where( input > 0, input, input*0.01)

class MLinear():
    def __init__(self, prev_neurons, current_neurons):
        self.weights = torch.randn(prev_neurons, current_neurons)
        self.bias = torch.zeros( (current_neurons,) )

    def forward(self, input ):
        return ( input @ self.weights ) + self.bias
    
class MNetwork():
    def __init__(self):
        self.layers = [
            MLinear(2,4),
            MLinear(4,4),
            MLinear(4,1)
        ]
        self.inter = []
    
    def forward(self, input):
        
        self.inter = [ {'a':input} ]
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
        loss = ( logits - targets) ** 2
        return loss.mean()
    
    def back(self, model, targets):

        b_grads = []
        w_grads = []

        c__a_p = 2 * ( model.inter[-1]['a'] - targets) / batch_size
        moving_gradient = c__a_p

        for idx in range(len(model.inter)-1, 0, -1 ):
            
            a__wsum_p = torch.where( model.inter[idx]['wsum']>0, 1.0, 0.01 )
            moving_gradient *= a__wsum_p

            b_grads.insert(0, moving_gradient.sum(dim=0) )

            wsum__w_p = model.inter[idx-1]['a']
            w_grads.insert(0,  wsum__w_p.T @ moving_gradient )

            wsum__a_p = model.layers[idx-1].weights
            moving_gradient = moving_gradient @ wsum__a_p.T
        return w_grads, b_grads

class AdamW():
    def __init__(self, model, b1=0.9, b2=0.99, eps=1e-8, weight_decay=0.01):
        self.t=0
        self.b1=b1
        self.b2=b2
        self.eps=eps
        self.weight_decay=weight_decay
        self.model = model

        self.w_mean = [ torch.zeros_like(l.weights) for l in model.layers ]
        self.w_varience = [ torch.zeros_like(l.weights) for l in model.layers ]

        self.b_mean = [ torch.zeros_like(l.bias) for l in model.layers  ]
        self.b_varience = [ torch.zeros_like(l.bias) for l in model.layers ]

    def step(self, w_grads, b_grads):

        self.t += 1
        for idx,l in enumerate(self.model.layers):
            # MOMENTUM's
            #MEAN
            self.w_mean[idx] = (self.b1 * self.w_mean[idx]) + ( (1-self.b1) * w_grads[idx]  )
            # VARIENCE
            self.w_varience[idx] = (self.b2 * self.w_varience[idx]) + ( (1-self.b2) * (w_grads[idx] ** 2)  )

            # CORRECTION
            w_mean_hat = self.w_mean[idx] / (1 - self.b1 ** self.t )
            w_varience_hat = self.w_varience[idx] / (1 - self.b2 ** self.t )

            # UPDATE
            l.weights -= lr * ( w_mean_hat / ( torch.sqrt(w_varience_hat)  + self.eps) + ( l.weights * self.weight_decay) )


            # BIAS
            # MOMENTUM's
            #MEAN
            self.b_mean[idx] = (self.b1 * self.b_mean[idx]) + ( (1-self.b1) * b_grads[idx]  )
            # VARIENCE
            self.b_varience[idx] = (self.b2 * self.b_varience[idx]) + ( (1-self.b2) * (b_grads[idx] ** 2)  )

            # CORRECTION
            b_mean_hat = self.b_mean[idx] / (1 - self.b1 ** self.t )
            b_varience_hat = self.b_varience[idx] / (1 - self.b2 ** self.t )

            # UPDATE
            l.bias -= lr * ( b_mean_hat / ( torch.sqrt(b_varience_hat)  + self.eps) + ( l.bias * self.weight_decay) )

        


def generate_xor(n_samples):
    q = torch.bernoulli(torch.full( (n_samples,2) , 0.5 )   )
    a =  (q[:,0] != q[:,1]).float()
    return q, a 

batch_size = 1
model = MNetwork()
optimizer = AdamW( model=model)
loss_obj = MSELoss()
x=[]
y=[]


lr=0.01

for i in range(15000):
    q, a = generate_xor(n_samples=batch_size)
    pred = model.forward(  q  )
    loss = loss_obj.get_loss(  logits = pred, targets = a.view(-1, pred.shape[-1] )  )
    w_grads , b_grads = loss_obj.back(model=model, targets = a.view(-1, pred.shape[-1] )   )
    optimizer.step(w_grads=w_grads, b_grads=b_grads)

    if i%100==0:
        print(f'step:{i}')

    x.append(i)
    y.append(loss.item())

plt.plot(x,y)
plt.show()



















if False: # OUTPUT FROM PYTORCH MODEL
    # TEST MODEL
    pt_model = torch.nn.Sequential(
        torch.nn.Linear(2,4),
        torch.nn.LeakyReLU(),

        torch.nn.Linear(4,4),
        torch.nn.LeakyReLU(),

        torch.nn.Linear(4,1),
        torch.nn.LeakyReLU()
    )

    with torch.no_grad():
        pt_model[0].weight.copy_(model.layers[0].weights.T)
        pt_model[0].bias.copy_(model.layers[0].bias)

        pt_model[2].weight.copy_(model.layers[1].weights.T)
        pt_model[2].bias.copy_(model.layers[1].bias)

        pt_model[4].weight.copy_(model.layers[2].weights.T)
        pt_model[4].bias.copy_(model.layers[2].bias)

    pt_model.zero_grad()
    pt_pred = pt_model.forward( q )
    pt_loss_fn = torch.nn.MSELoss(reduction='mean')
    pt_loss = pt_loss_fn( pt_pred,  a.view(-1, pt_pred.shape[-1] )  )
    pt_loss.backward()
    pt_w_grad = [
        pt_model[0].weight.grad.T,
        pt_model[2].weight.grad.T,
        pt_model[4].weight.grad.T,
    ]
    pt_b_grad = [
        pt_model[0].bias.grad,
        pt_model[2].bias.grad,
        pt_model[4].bias.grad,
    ]

    
    print(f'prediction:{pred}')
    print(f'pytorch pred:{pt_pred}')
    print(f' custom loss:{loss} pt_loss:{pt_loss}')

    print(f' custom grads: w:{w_grads} \n b:{b_grads}')
    print(f' autograd pytorch grads: w:{pt_w_grad} \n b:{pt_b_grad}')



