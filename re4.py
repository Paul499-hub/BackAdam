import torch

def relu(input):
    return torch.clamp(input, min=0.0)

class MLinear():
    def __init__(self, prev_neurons, current_neurons):
        self.weights = torch.randn(prev_neurons, current_neurons)
        self.bias = torch.ones((current_neurons,))

    def forward(self, input):
        return (input @ self.weights) + self.bias
    
class MNetwork():
    def __init__(self):
        self.layers = [
            MLinear(2,2),
            MLinear(2,2),
            MLinear(2,2)
        ]
        self.intermediates = None

    def forward(self, input):

        self.intermediates = [ {'activated':input}]
        out = input
        for l in self.layers:
            out = l.forward(out)
            self.intermediates.append( {'w_sum':out} )
            out = relu(out)
            self.intermediates[-1]['activated'] = out
            self.intermediates[-1]['weights'] = l.weights
        return out
    
class MSELoss():
    def __init__(self):
        self.targets = None

    def get_loss(self, logits, targets):
        self.targets = targets
        loss = (logits - targets) ** 2
        return loss.mean()
    
    def backward(self, intermediates):

        w_updates_saved=[]
        b_updates_saved=[]
        moving_gradient=None

        # FIRST STEP - MOVE GRADIENT TO A FROM COST 
        #print(f'---------- BACKPROPAGATION ---- \n\n')
        c__a_partial = 2 * ( intermediates[-1]['activated'] - self.targets)
        moving_gradient = c__a_partial
        #print(f'Gradient cost to a partial:{c__a_partial}')
        #print(f'Moving gradient:{c__a_partial}')
 
        # STEP FIVE REPEAT STEP 2 to step 4 
        for idx in range( len(intermediates)-1, 0, -1 ): # START: last index. STOP: BEFORE -1 so 0  STEP: -1 (DECREMENT BY ONE)
            #print(f'\n\n idx:{idx}')
            
            # SECOND STEP - MOVE GRADIENT FROM A to w_sum
            a__wsum_partial = torch.where( intermediates[idx]['w_sum'] > 0, torch.tensor(1.0), torch.tensor(0.0) )
            moving_gradient = moving_gradient * a__wsum_partial 
            #print(f'Gradient A to w_sum partial: {a__wsum_partial}')
            #print(f'Moving gradient: {moving_gradient}')

            # THIRD STEP - GET BIAS AND WEIGHTS GRADIENTS
            b_updates_saved.append(moving_gradient)

            # WEIGHTS
            wsum__w_partial = intermediates[idx-1]['activated']
            #print(f'wsum__w_partial:{wsum__w_partial}  moving_gradient:{moving_gradient}')
            w_updates_saved.append(   wsum__w_partial.unsqueeze(1) * moving_gradient.unsqueeze(0) )
            #print(f'w_updates_saved:{w_updates_saved}')

            # FOURTH STEP 
            wsum__a_partial = intermediates[idx]['weights']
            moving_gradient = moving_gradient @ wsum__a_partial.T 
            #print(f'moving gradient: {moving_gradient}')
        
        #print(f' ORIGINAL ORDER === {w_updates_saved} . REVERSED ORDER === {w_updates_saved[::-1]}')
        # list[start:stop:step]
        return w_updates_saved[::-1], b_updates_saved[::-1]
           
class SGD():
    def __init__(self):
        pass

    def step(self, w_grad, b_grad, model, lr):
        
        for idx in range(len(model.layers)):
            print(f' model layer [{idx}]: {model.layers[idx].weights }')
            print(f' weight gradients for this layer:{w_grad[idx]}')

model = MNetwork()

# WEIGHTS ARE COLUMN WISE (FIRST NEURON -> FIRST COLUMN... CONNECTIONS REPESENT CONNECTION TO PREVIOUS LAYER)
model.layers[0].weights = torch.tensor([[0.05, 0.06],
                                        [0.1,  0.11]]) 
model.layers[1].weights  = torch.tensor([[0.15, 0.16],
                                         [0.2, 0.21]]) 
model.layers[2].weights  = torch.tensor([[0.25, 0.26],
                                         [0.3, 0.31]]) 

loss_obj = MSELoss()
optimizer = SGD()

out = model.forward( torch.tensor([2.0, 5.0]) )
print(f'forward pass result:{out}')
loss = loss_obj.get_loss(out, torch.tensor([1.0, 3.0]) )
print(f'loss mean: {loss}')

w_grad, b_grad =  loss_obj.backward( intermediates = model.intermediates )
print(f'\n\n FINAL WEIGHT AND BIAS GRADIENTS(POSITIVE): WEIGHTS: {w_grad} \n BIAS:{b_grad}')

#optimizer.step( w_grad=w_grad, b_grad=b_grad, model=model, lr=0.001)



# GO THROUGH MODEL's WEIGHTS , MAKE A SMALL CHANGE(eps) TO A SINGLE WEIGHT , 
# CALCULATE LOSS(MEAN) 
# MAKE A SMALL NEGATIVE CHANGE(eps) TO SAME WEIGHT, THEN SUBTRACT THESE AND DIVIDE EVERYTHING BY 2eps

print( f'\n\n ++++++++++++++++++++++ NEW:{model.layers[-1].weights[0][0]}'  )

print(f' Gradients from Backpropagation:{w_grad}')

eps = 1e-4
p_grad_array = []
for i in range( len(model.layers) ):
    m_layer=model.layers[i]
    p_grad_array.append([])
    for j in range( len(m_layer.weights) ):
        m_layer_weights_row = model.layers[i].weights[j]
        p_grad_array[-1].append([])
        for k in range( len(m_layer_weights_row)):
            
            m_layer_weights_row_w= m_layer_weights_row[k]
            org_w = model.layers[i].weights[j][k].item()
            
            # POSITIVE PRETURBATION
            model.layers[i].weights[j][k] = model.layers[i].weights[j][k] + eps

            #CALCULATE LOSS
            pred = model.forward(torch.tensor([2.0, 5.0]))
            loss = loss_obj.get_loss( logits=pred, targets=torch.tensor([1.0, 3.0]) )
            
            # RESET WEIGHT VALUE TO OLD ONE
            model.layers[i].weights[j][k] = org_w

            # NEGATIVE PRETURBATION
            model.layers[i].weights[j][k] = model.layers[i].weights[j][k] - eps
            
            pred = model.forward(torch.tensor([2.0, 5.0]))
            loss2 = loss_obj.get_loss( logits=pred, targets=torch.tensor([1.0, 3.0]) )

            # CALCULATE  
            approx_grad = (loss - loss2) / (2 * eps)
            p_grad_array[-1][-1].append(approx_grad * 2 )
            #break
        #break
    #break
print(f'Preturbation tested gradients:{p_grad_array}')

            
            






