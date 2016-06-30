import theano
import theano.tensor as T
import theano.tensor.nnet as TN

##########
# gru cell
# return sequence output
##########
def gru_cell(x, y, E, V, U, W, b, c):
    
    def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
       
        # Word embedding layer
        x_e = E[:,x_t]
        
        # GRU Layer 1
        z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
        r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
        c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
        s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
        
        # GRU Layer 2
        z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
        r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
        c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
        s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
        
        # Final output calculation
        # Theano's softmax returns a matrix with one row, we only need the row
        o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

        return [o_t, s_t1, s_t2]
    
    [o, s, s2], updates = theano.scan(
        forward_prop_step,
        sequences=x,
        truncate_gradient=self.bptt_truncate,
        outputs_info=[None, 
                      dict(initial=T.zeros(self.hidden_dim)),
                      dict(initial=T.zeros(self.hidden_dim))])

    return o, x, y