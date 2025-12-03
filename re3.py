import torch

def relu(input):
    return torch.clamp(input, min=0.0)

class MLinear():
    def __init__(self, prev_neurons, current_neurons):
        self.weights = torch.randn(prev_neurons, current_neurons)
        torch.nn.init.kaiming_normal_(self.weights, mode='fan_in', nonlinearity='relu')
        self.bias = torch.ones((current_neurons,))

    def forward(self, input):
        return ( input @ self.weights ) + self.bias
    
class MNetwork():
    def __init__(self):
        self.layers = [
            MLinear(2,2),
            MLinear(2,2),
            MLinear(2,2)
        ]
        self.intermediate=[]

    def forward(self, input):

        self.intermediate=[] # <--- RESET INTERMEDIATE VALUES USED FOR BACKPROPAGATION

        out = input
        for l in self.layers:
            out = l.forward(out) # <--- EXPECTED INPUT (batch_size, features)
            self.intermediate.append({'w_sum':out})
            out = relu(out)
            self.intermediate[ -1 ]['activated'] = out
            self.intermediate[ -1 ]['weights'] = l.weights
        return out
    

class MSELoss():
    def __init__(self):
        self.targets = None

    def get_loss(self, logits, targets):
        self.targets = targets
        loss = (logits - targets ) ** 2 
        print(f'Loss (not mean):{loss}')
        return loss.mean() 
    
    def backward(self, intermediates ):

        b_updates_saved = [] 
        w_updates_saved = [] 
        moving_gradient = None

        # --------------------- FIRST BACKPROPAGATION STEP, MOVE GRADIEN FROM COST TO LAST LAYER ACTIVATION
        c__a = 2 * ( intermediates[-1]['activated'] - self.targets )
        moving_gradient = c__a
        print(f' cost partial gradient with respect to last layer activations: {c__a}')
        

        # --------------------- SECOND STEP, MOVE GRADIENT TO w_sum from ACTIVATION: 1(calculate activation partial gradient) 2 (multiply with last step's result)
        a__wsum = torch.where( intermediates[-1]['w_sum']>0, torch.tensor(1.0), torch.tensor(0.0) )
        moving_gradient = moving_gradient * a__wsum
        print(f' Partial gradient a-->w_sum :{a__wsum} . Moving gradient multipled with this result: {moving_gradient}')

        # ---------------------  THIRD STEP, MOVE GRADIENT TO BIAS, save bias gradients for future updates. SAME WITH WEIGHTS
        #BIAS PARTIAL DERIVATIVE IS ALWAYS * 1 because the derivative of adding a scalar is always 1. WE DONT NEED TO COMPUTE ANYTHING HERE, JUST SAVE 
        b_updates_saved.append(moving_gradient)

        #WEIGHT GRADIENT IS moving_gradient * previous_layer activation result 
        # WE NEED prev_layer_neurons * current_layer_neurons ammount of updates, so we have to create dimensions for matrix element wise multiplication. 
        # THIS IS ELEMENT WISE MULTIPLICATION 1,2 x 2,1 and BROADCASTING CREATES COPIES resulting in [2,2] final result of ELEMENT WISE MULTIPLICATION x1y1 x1y2 NEW ROW x2y1 x2y2
        w_updates_saved.append( moving_gradient.unsqueeze(0) * intermediates[-2]['activated'].unsqueeze(1) )
        print(f''' Partial gradient w_sum-->w :{intermediates[-2]["activated"] }. 
              Moving gradient moved to w (will be used to update weights): { moving_gradient.unsqueeze(0) * intermediates[-2]["activated"].unsqueeze(1) }''')
        
        # ---------------------  FOURTH STEP , CALCULATE w_sum partial derivative with respect to Previous activations, MOVE GRADIENT TO NEXT (-1) LAYER a 
        # WE NEED A TRANSPOSE TO SUM THE PARTIAL GRADIENTS(w) TO CORRECT NEURONS 
        wsum__a = intermediates[-1]['weights']
        moving_gradient = moving_gradient @ wsum__a.T  # shape [2] @ [2,2]
        print(f'Gradients at (a)(-2): {moving_gradient}')
        
        # ------------------- STEP FIVE WILL BE LOOP FROM STEP TWO TO STEP FOUR BUT CHANGE LAYER INDICES 


model = MNetwork()
loss_obj = MSELoss()

# WEIGHTS ARE COLUMN WISE (FIRST NEURON -> FIRST COLUMN... CONNECTIONS REPESENT CONNECTION TO PREVIOUS LAYER)
model.layers[0].weights = torch.tensor([[0.05, 0.06],
                                        [0.1,  0.11]]) 
model.layers[1].weights  = torch.tensor([[0.15, 0.16],
                                         [0.2, 0.21]]) 
model.layers[2].weights  = torch.tensor([[0.25, 0.26],
                                         [0.3, 0.31]]) 

out = model.forward( torch.tensor([2.0,5.0]) )
print(f' MODEL INTERMEDIATES:{model.intermediate}')
loss = loss_obj.get_loss(logits=out, targets= torch.tensor([1.0, 3.0]))
print(f'loss mean:{loss}')
loss_obj.backward( intermediates = model.intermediate )




    
