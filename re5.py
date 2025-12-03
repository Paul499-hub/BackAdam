import torch
import random

def relu(input):
    return torch.clamp(input, min=0.0)

def leaky_relu(x, alpha=0.01):
    return torch.where(x>0, x, alpha * x)

class MLinear():
    def __init__(self, prev_neurons, current_neurons):
        #self.weights = torch.randn(prev_neurons, current_neurons)
        #self.weights = torch.empty(prev_neurons, current_neurons)
        self.weights = torch.randn(prev_neurons, current_neurons)

        #self.bias = torch.ones((current_neurons,))
        self.bias = torch.zeros((current_neurons,))
        
        # torch.nn.init.kaiming_normal_(
        #     self.weights,
        #     mode='fan_in',
        #     nonlinearity='leaky_relu',
        #     a=0.01
        # )

    def forward(self, input):
        return (input @ self.weights ) + self.bias
    
class MNetwork():
    def __init__(self):
        self.layers = [
            MLinear(2,2),
            MLinear(2,2),
            MLinear(2,2),
        ]
        self.intermediates = None

    def forward(self, input):

        self.intermediates = [{'activated':input}]
        out = input
        for i in range(len(self.layers)) :
            l = self.layers[i]
            out = l.forward(out)
            self.intermediates.append({'w_sum':out})
            #out = relu(out)
            out = leaky_relu(out)
            self.intermediates[-1]['activated'] = out
            self.intermediates[-1]['weights'] = l.weights

        return out
    
class MSELoss():
    def __init__(self):
        pass

    def get_loss(self, logits, targets):
        loss = (logits - targets) ** 2
        return loss.mean()
    
    def back(self, intermediates, targets):
        
        b_gradient_saved = []
        w_gradient_saved = []
        moving_gradient = None

        c__a_p = 2 * (intermediates[-1]['activated'] - targets ) / targets.numel()
        moving_gradient = c__a_p

        for idx in range( len(intermediates)-1, 0 , -1 ):

            #a__wsum_p = torch.where( intermediates[idx]['w_sum']>0, 1.0, 0.0) # RELU DERIVATIVE
            a__wsum_p = torch.where( intermediates[idx]['w_sum'] > 0, 1.0, 0.01 ) # Leaky RELU DERIVATIVE
            moving_gradient = moving_gradient * a__wsum_p

            b_gradient_saved.append(moving_gradient)

            wsum__w_p = intermediates[idx-1]['activated']
            # w_gradient_saved.append(  moving_gradient.unsqueeze(0) * wsum__w_p.unsqueeze(1) )
            w_gradient_saved.append( wsum__w_p[:, None] * moving_gradient[None, :] ) 
   
            wsum__a_p = intermediates[idx]['weights']
            moving_gradient = moving_gradient @ wsum__a_p.T
        
        return w_gradient_saved[::-1] , b_gradient_saved[::-1]

class SGD():
    def __init__(self):
        pass

    def step(self, model, w_grad_updates, b_grad_updates, lr):
        for i in range( len(model.layers )):
            model.layers[i].weights -= w_grad_updates[i] * lr
            model.layers[i].bias -= b_grad_updates[i] * lr
        

def preturbation_approx_grad(model, loss_obj):
    eps = 1e-4
    a_grad_array= []
    a_b_grad_array = []
    for i in range( len(model.layers) ):
        for j in range( len(model.layers[i].weights)):
            for k in range( len(model.layers[i].weights[j])):
            
                saved_w = model.layers[i].weights[j][k].item()

                model.layers[i].weights[j][k] = model.layers[i].weights[j][k] + eps
                pred = model.forward( input=torch.tensor([2.0, 5.0]))
                loss = loss_obj.get_loss( logits=pred, targets= torch.tensor([1.0, 3.0]))

                model.layers[i].weights[j][k] = saved_w

                model.layers[i].weights[j][k] = model.layers[i].weights[j][k] - eps
                pred = model.forward( input=torch.tensor([2.0, 5.0]))
                loss2 = loss_obj.get_loss( logits=pred, targets= torch.tensor([1.0, 3.0]))

                approx_grad = (loss - loss2) / (2 * eps)
                a_grad_array.append(approx_grad)

                model.layers[i].weights[j][k] = saved_w

    for i in range( len(model.layers)):
        for j in range( len(model.layers[i].bias ) ):
            
            saved_b = model.layers[i].bias[j].item()

            # + PRETURB BIAS
            model.layers[i].bias[j] = model.layers[i].bias[j] + eps

            # GET LOSS
            pred = model.forward( input=torch.tensor([2.0, 5.0]))
            loss = loss_obj.get_loss( logits=pred, targets=torch.tensor([1.0, 3.0])) 

            # RESET
            model.layers[i].bias[j] = saved_b

            # - PRETURB BIAS
            model.layers[i].bias[j] = model.layers[i].bias[j] - eps

            # GET SECOND LOSS
            pred = model.forward(input=torch.tensor([2.0, 5.0]))
            loss2 = loss_obj.get_loss( logits=pred, targets=torch.tensor([1.0, 3.0]))

            # CALCULATE APPROX GRADIENT FOR BIAS
            approx_grad = (loss - loss2) / (2 * eps)
            a_b_grad_array.append(approx_grad)

            #RESET 
            model.layers[i].bias[j] = saved_b

                
    print(f' Approx weight gradients: {a_grad_array}')
    print(f'\n Apporx bias gradients: {a_b_grad_array}')

def monitor_weight_magnitudes(model):
    for i in range(len(model.layers)):
        print(f'Model layers weight mean:{model.layers[i].weights.abs().mean()}')

def generate_xor_sample(n_samples):
    prob = 0.5
    samples = torch.bernoulli( torch.full((n_samples,2), prob))
    ground_truth = ( samples[:, 0] != samples[:, 1] ).float()
    return samples, ground_truth
    
def infer():

    while True:
        i1 = float(int(input('\n\n======= Enter 1 or 0 : ')))
        i2 = float(int(input('======= Enter 1 or 0 : ')))
        pred = model.forward(input=torch.tensor([i1, i2]))
        loss= loss_obj.get_loss(logits=pred, targets= torch.tensor( 1.0 if i1!=i2 else 0.0 )  )
        print(f'-->  [O__O] Prediction : {pred} \n')
        print(f'------- True output : { 1.0 if i1!=i2 else 0.0 }')
        print(f'------- Loss : {loss}')
        
        if input('break?(y/n):')=='y':
            break


model = MNetwork()
loss_obj = MSELoss()
optimizer= SGD()
lr = 0.01
print(f'\n\n\n\n HELLO WORLD__________________')
xor_dataset, answer_dataset = generate_xor_sample(n_samples=3)
#print(f'xor dataset:{xor_dataset} \n answer dataset:{answer_dataset}')

# # Example: Precompute dataset statistics
# data = torch.tensor([[2.0, 5.0], [1.0, 3.0], ...])  # All your data
# data_mean = data.mean(dim=0)  # Per-feature mean: tensor([mean_x1, mean_x2])
# data_std = data.std(dim=0)    # Per-feature std:  tensor([std_x1, std_x2])

# # Normalize a single input
# input = torch.tensor([2.0, 5.0])
# input_normalized = (input - data_mean) / data_std


# for step in range( len(xor_dataset)):
#     pred = model.forward( xor_dataset[step] )
#     loss = loss_obj.get_loss( logits=pred, targets = answer_dataset[step] )
#     w_grad , b_grad = loss_obj.back( intermediates=model.intermediates, targets=answer_dataset[step] )

#     print(f'\n\nInput:{xor_dataset[step]}')
#     print(f'Expected output:{answer_dataset[step]}')
#     print(f'-->  [O__O] Prediction:{pred}')
#     print(f'-->  Loss:{loss}')
#     print(f'w gradients:{w_grad}')
#     print(f'b_gradients:{b_grad} \n\n')
#     preturbation_approx_grad( model=model, loss_obj=loss_obj )

#     optimizer.step(model=model, w_grad_updates=w_grad, b_grad_updates=b_grad, lr=lr )
    #monitor_weight_magnitudes(model=model)



# OLD TRAINING LOOP 
for step in range( 150 ):
    pred = model.forward( torch.tensor([5.0, 1.0]) )
    loss = loss_obj.get_loss( logits=pred, targets = torch.tensor([1.0, 3.0]) )
    w_grad , b_grad = loss_obj.back( intermediates=model.intermediates, targets=torch.tensor([1.0, 3.0]) )

    # print(f'\n\nInput:{xor_dataset[step]}')
    # print(f'Expected output:{answer_dataset[step]}')
    print(f'step {step}-->  [O__O] Prediction:{pred}')
    print(f'-->  Loss:{loss}')
    # print(f'w gradients:{w_grad}')
    # print(f'b_gradients:{b_grad} \n\n')
    # preturbation_approx_grad( model=model, loss_obj=loss_obj )

    optimizer.step(model=model, w_grad_updates=w_grad, b_grad_updates=b_grad, lr=lr )

print(f'w gradients:{w_grad}')
print(f'b_gradients:{b_grad} \n\n')
preturbation_approx_grad( model=model, loss_obj=loss_obj )

    # infer()











