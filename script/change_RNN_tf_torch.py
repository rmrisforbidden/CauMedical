class RNN():
    rnn1 = torch.nn.GRU(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=n_layers,
                         bidirectional=False,
                         bias=True,
                         batch_first=True).cuda()


    def forward(x):
        output1, hn1 = rnn1(x, h0)
        output2, hn2 = rnn1(hn1, h0)
        output3, hn3 = rnn1(hn2, h0)

        x1 = gru1(inputs, initial_state=h0);
        x2 = gru2(x1);
        x3 = gru3(x2);
        x4 = tf.keras.layers.Dense(output_size,activation="linear")(x3)

def change_keras_to_torch():
    # Load model
    # Covert weight, input, bias

    # Put in the torch model


class GRU(nn.Module):
    def __init__(self,input_size=5, hidden_size_1=32, hidden_size_2=32, hidden_size_3=32, output_size=3, num_layers=1, device='cuda'):
        super(GRU, self).__init__()
        """
        # This model is trained in the paper: 
        # output : (m, dm/dT1, dm/dT2)
        # input : (RF, T1, T2, TE, TR)
        """
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.num_layers = num_layers
        self.device = device
        
        self.gru_1 = nn.GRU(input_size, hidden_size_1, num_layers, batch_first=True)
        self.gru_2 = nn.GRU(hidden_size_1, hidden_size_2, num_layers, batch_first=True)
        self.gru_3 = nn.GRU(hidden_size_1, hidden_size_3, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size_3, output_size)

    def forward(self, x, h_set):
        input_X = x
        out_gru_1 ,h_set[0] = self.gru_1(input_X, h_set[0])   # h_1
        out_gru_2 ,h_set[1] = self.gru_2(out_gru_1, h_set[1]) # h_2
        out_gru_3 ,h_set[2] = self.gru_2(out_gru_2, h_set[2]) # h_3
        out_Dense_out = self.fc_out(out_gru_3) 

        return out_Dense_out, h_set
    
    def init_hidden_set(self):
        h_set = {}
        h_set[0] = torch.zeros(self.num_layers, self.hidden_size_1, device=self.device) # h_1
        h_set[1] = torch.zeros(self.num_layers, self.hidden_size_2, device=self.device) # h_2
        h_set[2] = torch.zeros(self.num_layers, self.hidden_size_3, device=self.device) # h_3
        return h_set



"""# 1) Define RNN layer with 3 stacked GRU units"""
import tensorflow as tf
def build_model_GRU_init(rnn_units, input_batch, input_size, output_size):

  inputs = tf.keras.Input(batch_size = input_batch, shape=[None, input_size],name='input_1');
  init_state = tf.keras.Input(batch_size = input_batch, shape = [3], name='input_2');
  h0 = tf.keras.layers.Dense(rnn_units,activation="linear")(init_state)

  gru1 = tf.keras.layers.GRU(rnn_units,
              return_sequences=True,
              stateful=False, ## See definition of the stateful input, 
              recurrent_initializer='glorot_uniform',unroll=False);
  gru2 = tf.keras.layers.GRU(rnn_units,
              return_sequences=True,
              stateful=False, ## See definition of the stateful input, 
              recurrent_initializer='glorot_uniform',unroll=False);
  gru3 = tf.keras.layers.GRU(rnn_units,
              return_sequences=True,
              stateful=False, ## See definition of the stateful input, 
              recurrent_initializer='glorot_uniform',unroll=False);

  x1 = gru1(inputs, initial_state=h0);
  x2 = gru2(x1);
  x3 = gru3(x2);
  x4 = tf.keras.layers.Dense(output_size,activation="linear")(x3)
  return tf.keras.Model(inputs = [inputs, init_state], outputs = x4, name='GRU')


def convert_input_kernel(kernel):
    kernel_z, kernel_r, kernel_h = np.hsplit(kernel, 3)
    kernels = [kernel_r, kernel_z, kernel_h]
    return np.vstack([k.reshape(k.T.shape) for k in kernels])

def convert_recurrent_kernel(kernel):
    kernel_z, kernel_r, kernel_h = np.hsplit(kernel, 3)
    kernels = [kernel_r, kernel_z, kernel_h]
    return np.vstack(kernels)

def convert_bias(bias):
    bias = bias.reshape(2, 3, -1) 
    return bias[:, [1, 0, 2], :].reshape(-1)