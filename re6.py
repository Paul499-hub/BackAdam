import torch
import copy
import matplotlib.pyplot as plt

def l_relu(input):
    return torch.where( input > 0, input, input * 0.01 )


class MLinear():
    def __init__(self, prev_neurons, current_neurons):
        #self.weights = torch.randn(prev_neurons, current_neurons)
        #Xavier initialization
        limit = torch.sqrt(torch.tensor(6.0 / (prev_neurons + current_neurons)))
        self.weights = torch.empty( prev_neurons, current_neurons).uniform_(-limit, limit)
        self.bias = torch.zeros((current_neurons,))

    def forward(self, input):
        return (input @ self.weights ) + self.bias 
    
class MNetwork():
    def __init__(self):
        self.layers = [
            MLinear(2,4),
            MLinear(4,2),
            MLinear(2,1)
        ]
        self.intermediates = None

    def forward(self, x):
        
        # SAVE INPUTS AS FIRST LAYER OF NEURONS FOR BACKPROP
        self.intermediates= [{'activated':x}]
        for l in self.layers:
            x = l.forward(x)
            self.intermediates.append({'w_sum':x})
            x = l_relu(x)
            self.intermediates[-1]['activated'] = x
            self.intermediates[-1]['weights'] = l.weights
        return x
    
class MSELoss():
    def __init__(self):
        pass

    def get_loss(self, logits, targets):
        loss = (logits - targets) ** 2
        return loss.mean()
    

    def back(self , model, targets):
        
        moving_gradient = None
        b_grads = []
        w_grads = []

        c__a_p =  2 * ( model.intermediates[-1]['activated'] - targets )
        moving_gradient = c__a_p

        for idx in range( len(model.intermediates)-1, 0, -1 ):
            
            a__wsum_p = torch.where(  model.intermediates[idx]['w_sum'] > 0 , 1.0, 0.01 )  
            moving_gradient *= a__wsum_p
            b_grads.append( moving_gradient )
            w_grads.append(  moving_gradient.unsqueeze(0) * model.intermediates[idx-1]['activated'].unsqueeze(1) )
            moving_gradient = moving_gradient @  model.intermediates[idx]['weights'].T 
        return  w_grads[::-1] , b_grads[::-1]
 


    
class SGD():
    def __init__(self):
        pass

    def step(self, model, w_grad, b_grad):

        for i in range(len(model.layers)):
            model.layers[i].weights -= w_grad[i] * lr
            model.layers[i].bias -= b_grad[i] * lr


def approx_gradients(g_model, loss_obj):
    model = copy.deepcopy(g_model)
    eps = 1e-4
    approx_w_grad_array = []
    approx_b_grad_array = []
    for i in range(len(model.layers)):
        for j in range(len(model.layers[i].weights)):
            for k in range( len(model.layers[i].weights[j]) ):
                #print(f'Models weight:{model.layers[i].weights[j, k]}')
                w_saved = model.layers[i].weights[j, k].item()
                
                # GET + preturbed LOSS
                model.layers[i].weights[j, k] += eps
                pred = model.forward( inp )
                loss1 = loss_obj.get_loss( logits = pred, targets = ans)

                # RESET
                model.layers[i].weights[j, k] = w_saved

                # GET - preturbed LOSS
                model.layers[i].weights[j, k] -= eps
                pred = model.forward( inp )
                loss2 = loss_obj.get_loss( logits = pred, targets = ans)

                # CALCULATE Finite differences approximated gradient for this weight
                approx_w_grad = (loss1 - loss2) / (2 * eps)
                approx_w_grad_array.append( approx_w_grad.item() * 2 )

                # RESET
                model.layers[i].weights[j, k] = w_saved

        # NOW SAME FOR BIAS
        for m in range( len(model.layers[i].bias)):
            #print(f'org bias:{model.layers[i].bias[m]}')
            b_saved = model.layers[i].bias[m].item()

            # + preturbed loss
            model.layers[i].bias[m] += eps
            pred = model.forward( inp )
            b_loss1 = loss_obj.get_loss( logits = pred, targets = ans)

            # RESET 
            model.layers[i].bias[m] = b_saved 

            # - preturbed loss
            model.layers[i].bias[m] -= eps
            pred = model.forward( inp )
            b_loss2 = loss_obj.get_loss( logits = pred, targets = ans)

            # CALCULATE APPROX b GRAD
            approx_b_grad = (b_loss1 - b_loss2) / (2 * eps)
            approx_b_grad_array.append( approx_b_grad.item() * 2   )

            # RESET 
            model.layers[i].bias[m] = b_saved 

    return approx_w_grad_array, approx_b_grad_array

def generate_xor_samples(n_samples):
    prob = 0.5
    q = torch.bernoulli(torch.full((n_samples,2), prob)) 
    a = (q[:,0] != q[:,1]).float() 
    return q, a

def infer(model):
    while True:
        i1 = float(input('Enter 1 or 0 : '))
        i2 = float(input('Enter 1 or 0 : '))
        print(f'Expecting answer: {  float(i1 != i2) }')
        pred = model.forward( x = torch.tensor([i1, i2]) )
        print(f' [ O __ O ]: {pred}')

        if input(f'break?(y/n)') == 'y':
            break
    

model = MNetwork()
loss_obj = MSELoss()
optimizer = SGD()
plt_x,plt_y = [],[]
fig,ax = plt.subplots()


# inp = torch.tensor([1.0, 5.0])1
# ans = torch.tensor([ 1.0, 3.0])
lr = 0.01
q_dataset, a_dataset = generate_xor_samples(n_samples=32000)

for i in range( len(q_dataset)):

    pred = model.forward( x = q_dataset[i] )
    loss = loss_obj.get_loss( logits = pred, targets = a_dataset[i] )
    
    if False: # PRINT q a NETWORK RESPONSE and loss
        print(f'\n\nq:{q_dataset[i]}')
        print(f'a:{a_dataset[i]}')
        print(f' [O__O]: {pred}')
        print(f' loss:{ loss }')
    
    plt_x.append(i)
    plt_y.append(loss)
    
    #optimizer.step( model=model, w_grad=a_g_a, b_grad=a_b_g_a )
    w_grad, b_grad = loss_obj.back( model=model, targets= a_dataset[i] )

    if False: # PRINT AND COMPARE TO APPROXIMATED GRADIENTS
        a_g_a, a_b_g_a = approx_gradients(g_model=model, loss_obj=loss_obj)
        print(f'[BACKPROP]\n w_grad:{w_grad}\n [FINITE DIFFERENCE APPROXIMATIONS] \n {a_g_a} \n\n')
        print(f'[B BACKPROP]\n w_grad:{b_grad}\n [B FINITE DIFFERENCE APPROXIMATIONS] \n {a_b_g_a}')
        print(f'\n OPTIMIZER STEP COMPLETE...')

    optimizer.step( model=model, w_grad=w_grad, b_grad=b_grad)

plt.plot(plt_x, plt_y)
plt.show()
# infer(model=model)



