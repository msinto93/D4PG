'''
## Network ##
# Defines the D4PG Value (critic) and Policy (Actor) networks - with and without batch norm
@author: Mark Sinton (msinto93@gmail.com) 
'''

import tensorflow as tf
import numpy as np
from utils.ops import dense, relu, tanh, batchnorm, softmax
from utils.l2_projection import _l2_project

class Critic:
    def __init__(self, state, action, state_dims, action_dims, dense1_size, dense2_size, final_layer_init, num_atoms, v_min, v_max, scope='critic'):
        # state - State input to pass through the network
        # action - Action input for which the Z distribution should be predicted
         
        self.state = state
        self.action = action
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.scope = scope    
         
        with tf.variable_scope(self.scope):           
            self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
                         
            self.dense1 = relu(self.dense1_mul, scope='dense1')
             
            #Merge first dense layer with action input to get second dense layer            
            self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2a')        
             
            self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2b') 
                           
            self.dense2 = relu(self.dense2a + self.dense2b, scope='dense2')
                          
            self.output_logits = dense(self.dense2, num_atoms, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                       bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output_logits')  
            
            self.output_probs = softmax(self.output_logits, scope='output_probs')
                         
                          
            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [] # No batch norm params
            
            
            self.z_atoms = tf.lin_space(v_min, v_max, num_atoms)
            
            self.Q_val = tf.reduce_sum(self.z_atoms * self.output_probs) # the Q value is the mean of the categorical output Z-distribution
          
            self.action_grads = tf.gradients(self.output_probs, self.action, self.z_atoms) # gradient of mean of output Z-distribution wrt action input - used to train actor network, weighing the grads by z_values gives the mean across the output distribution
            

    def train_step(self, target_Z_dist, target_Z_atoms, IS_weights, learn_rate, l2_lambda):
        # target_Z_dist - target Z distribution for next state
        # target_Z_atoms - atom values of target network with Bellman update applied
         
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)               
                
                # Project the target distribution onto the bounds of the original network
                target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.z_atoms)  
                
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits, labels=tf.stop_gradient(target_Z_projected))
                self.weighted_loss = self.loss * IS_weights
                self.mean_loss = tf.reduce_mean(self.weighted_loss)
                                                
                self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.mean_loss + self.l2_reg_loss
                 
                train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)
                  
                return train_step
        

class Actor:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size, final_layer_init, scope='actor'):
        # state - State input to pass through the network
        # action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space
         
        self.state = state
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.scope = scope
         
        with tf.variable_scope(self.scope):
                    
            self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
                         
            self.dense1 = relu(self.dense1_mul, scope='dense1')
             
            self.dense2_mul = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))), scope='dense2')        
                         
            self.dense2 = relu(self.dense2_mul, scope='dense2')
             
            self.output_mul = dense(self.dense2, self.action_dims, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output') 
             
            self.output_tanh = tanh(self.output_mul, scope='output')
             
            # Scale tanh output to lower and upper action bounds
            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))
             
            
            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [] # No batch norm params
        
        
    def train_step(self, action_grads, learn_rate, batch_size):
        # action_grads - gradient of value output wrt action from critic network
         
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                 
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)  
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
                 
                train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))
                 
                return train_step
            
            
class Critic_BN:
    def __init__(self, state, action, state_dims, action_dims, dense1_size, dense2_size, final_layer_init, num_atoms, v_min, v_max, is_training=False, scope='critic'):
        # state - State input to pass through the network
        # action - Action input for which the Z distribution should be predicted
        
        self.state = state
        self.action = action
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.is_training = is_training
        self.scope = scope    

        
        with tf.variable_scope(self.scope):
            self.input_norm = batchnorm(self.state, self.is_training, scope='input_norm')
           
            self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
            
            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')
            
            self.dense1 = relu(self.dense1_bn, scope='dense1')
            
            #Merge first dense layer with action input to get second dense layer            
            self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2a')        
            
            self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2b') 
            
            self.dense2 = relu(self.dense2a + self.dense2b, scope='dense2')
            
            self.output_logits = dense(self.dense2, num_atoms, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                       bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output_logits')  
            
            self.output_probs = softmax(self.output_logits, scope='output_probs')
                         
                          
            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [v for v in tf.global_variables(scope=self.scope) if 'batch_normalization/moving' in v.name]
            
            
            self.z_atoms = tf.lin_space(v_min, v_max, num_atoms)
            
            self.Q_val = tf.reduce_sum(self.z_atoms * self.output_probs) # the Q value is the mean of the categorical output Z-distribution
          
            self.action_grads = tf.gradients(self.output_probs, self.action, self.z_atoms) # gradient of mean of output Z-distribution wrt action input - used to train actor network, weighing the grads by z_values gives the mean across the output distribution
            
    def train_step(self, target_Z_dist, target_Z_atoms, IS_weights, learn_rate, l2_lambda):
        # target_Z_dist - target Z distribution for next state
        # target_Z_atoms - atom values of target network with Bellman update applied
         
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(learn_rate)               
                
                # Project the target distribution onto the bounds of the original network
                target_Z_projected = _l2_project(target_Z_atoms, target_Z_dist, self.z_atoms)  
                
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits, labels=tf.stop_gradient(target_Z_projected))
                self.weighted_loss = self.loss * IS_weights
                self.mean_loss = tf.reduce_mean(self.weighted_loss)
                                                
                self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.network_params if 'kernel' in v.name]) * l2_lambda
                self.total_loss = self.mean_loss + self.l2_reg_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope) # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.minimize(self.total_loss, var_list=self.network_params)
                 
                return train_step
        

class Actor_BN:
    def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size, final_layer_init, is_training=False, scope='actor'):
        # state - State input to pass through the network
        # action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space
        
        self.state = state
        self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
        self.action_dims = np.prod(action_dims)
        self.action_bound_low = action_bound_low
        self.action_bound_high = action_bound_high
        self.is_training = is_training
        self.scope = scope
        
        with tf.variable_scope(self.scope):
        
            self.input_norm = batchnorm(self.state, self.is_training, scope='input_norm')
           
            self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
            
            self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')
            
            self.dense1 = relu(self.dense1_bn, scope='dense1')
            
            self.dense2_mul = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))),
                                bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))), scope='dense2')        
            
            self.dense2_bn = batchnorm(self.dense2_mul, self.is_training, scope='dense2')
            
            self.dense2 = relu(self.dense2_bn, scope='dense2')
            
            self.output_mul = dense(self.dense2, self.action_dims, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
                                bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output') 
            
            self.output_tanh = tanh(self.output_mul, scope='output')
            
            # Scale tanh output to lower and upper action bounds
            self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))
            
           
            self.network_params = tf.trainable_variables(scope=self.scope)
            self.bn_params = [v for v in tf.global_variables(scope=self.scope) if 'batch_normalization/moving' in v.name]
        
    def train_step(self, action_grads, learn_rate, batch_size):
        # action_grads - gradient of value output wrt action from critic network
        
        with tf.variable_scope(self.scope):
            with tf.variable_scope('train'):
                
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
                self.grads = tf.gradients(self.output, self.network_params, -action_grads)  
                self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope) # Ensure batch norm moving means and variances are updated every training step
                with tf.control_dependencies(update_ops):
                    train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))
                
                return train_step
    
    
    