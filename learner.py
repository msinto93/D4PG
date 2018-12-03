'''
## Learner ##
# Learner class - this trains the D4PG network on experiences sampled (by priority) from the PER buffer
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import sys
import tensorflow as tf
import numpy as np

from params import train_params
from utils.network import Actor, Actor_BN, Critic, Critic_BN
                    
class Learner:
    def __init__(self, sess, PER_memory, run_agent_event, stop_agent_event):
        print("Initialising learner... \n\n")
        
        self.sess = sess
        self.PER_memory = PER_memory
        self.run_agent_event = run_agent_event
        self.stop_agent_event = stop_agent_event
        
    def build_network(self):
        
        # Define input placeholders    
        self.state_ph = tf.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + train_params.STATE_DIMS))
        self.action_ph = tf.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + train_params.ACTION_DIMS))
        self.target_atoms_ph = tf.placeholder(tf.float32, (train_params.BATCH_SIZE, train_params.NUM_ATOMS)) # Atom values of target network with Bellman update applied
        self.target_Z_ph = tf.placeholder(tf.float32, (train_params.BATCH_SIZE, train_params.NUM_ATOMS))  # Future Z-distribution - for critic training
        self.action_grads_ph = tf.placeholder(tf.float32, ((train_params.BATCH_SIZE,) + train_params.ACTION_DIMS)) # Gradient of critic's value output wrt action input - for actor training
        self.weights_ph = tf.placeholder(tf.float32, (train_params.BATCH_SIZE)) # Batch of IS weights to weigh gradient updates based on sample priorities

        # Create value (critic) network + target network
        if train_params.USE_BATCH_NORM:
            self.critic_net = Critic_BN(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, is_training=True, scope='learner_critic_main')
            self.critic_target_net = Critic_BN(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, is_training=True, scope='learner_critic_target')
        else:
            self.critic_net = Critic(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, scope='learner_critic_main')
            self.critic_target_net = Critic(self.state_ph, self.action_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, train_params.NUM_ATOMS, train_params.V_MIN, train_params.V_MAX, scope='learner_critic_target')
        
        # Create policy (actor) network + target network
        if train_params.USE_BATCH_NORM:
            self.actor_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=True, scope='learner_actor_main')
            self.actor_target_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=True, scope='learner_actor_target')
        else:
            self.actor_net = Actor(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope='learner_actor_main')
            self.actor_target_net = Actor(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope='learner_actor_target')
     
        # Create training step ops
        self.critic_train_step = self.critic_net.train_step(self.target_Z_ph, self.target_atoms_ph, self.weights_ph, train_params.CRITIC_LEARNING_RATE, train_params.CRITIC_L2_LAMBDA)
        self.actor_train_step = self.actor_net.train_step(self.action_grads_ph, train_params.ACTOR_LEARNING_RATE, train_params.BATCH_SIZE)
        
        # Create saver for saving model ckpts (we only save learner network vars)
        model_name = train_params.ENV + '.ckpt'
        self.checkpoint_path = os.path.join(train_params.CKPT_DIR, model_name)        
        if not os.path.exists(train_params.CKPT_DIR):
            os.makedirs(train_params.CKPT_DIR)
        saver_vars = [v for v in tf.global_variables() if 'learner' in v.name]
        self.saver = tf.train.Saver(var_list = saver_vars, max_to_keep=201) 
        
    def build_update_ops(self):     
        network_params = self.actor_net.network_params + self.critic_net.network_params
        target_network_params = self.actor_target_net.network_params + self.critic_target_net.network_params
        
        # Create ops which update target network params with hard copy of main network params
        init_update_op = []
        for from_var,to_var in zip(network_params, target_network_params):
            init_update_op.append(to_var.assign(from_var))
        
        # Create ops which update target network params with fraction of (tau) main network params
        update_op = []
        for from_var,to_var in zip(network_params, target_network_params):
            update_op.append(to_var.assign((tf.multiply(from_var, train_params.TAU) + tf.multiply(to_var, 1. - train_params.TAU))))        
            
        self.init_update_op = init_update_op
        self.update_op = update_op
    
    def initialise_vars(self):
        # Load ckpt file if given, otherwise initialise variables and hard copy to target networks
        if train_params.CKPT_FILE is not None:
            #Restore all learner variables from ckpt
            ckpt = train_params.CKPT_DIR + '/' + train_params.CKPT_FILE
            ckpt_split = ckpt.split('-')
            step_str = ckpt_split[-1]
            self.start_step = int(step_str)    
            self.saver.restore(self.sess, ckpt)
        else:
            self.start_step = 0
            self.sess.run(tf.global_variables_initializer())   
            # Perform hard copy (tau=1.0) of initial params to target networks
            self.sess.run(self.init_update_op)
            
    def run(self):
        # Sample batches of experiences from replay memory and train learner networks 
            
        # Initialise beta to start value
        priority_beta = train_params.PRIORITY_BETA_START
        beta_increment = (train_params.PRIORITY_BETA_END - train_params.PRIORITY_BETA_START) / train_params.NUM_STEPS_TRAIN
        
        # Can only train when we have at least batch_size num of samples in replay memory
        while len(self.PER_memory) <= train_params.BATCH_SIZE:
            sys.stdout.write('\rPopulating replay memory up to batch_size samples...')   
            sys.stdout.flush()
        
        # Training
        sys.stdout.write('\n\nTraining...\n')   
        sys.stdout.flush()
    
        for train_step in range(self.start_step+1, train_params.NUM_STEPS_TRAIN+1):  
            # Get minibatch
            minibatch = self.PER_memory.sample(train_params.BATCH_SIZE, priority_beta) 
            
            states_batch = minibatch[0]
            actions_batch = minibatch[1]
            rewards_batch = minibatch[2]
            next_states_batch = minibatch[3]
            terminals_batch = minibatch[4]
            gammas_batch = minibatch[5]
            weights_batch = minibatch[6]
            idx_batch = minibatch[7]            
    
            # Critic training step    
            # Predict actions for next states by passing next states through policy target network
            future_action = self.sess.run(self.actor_target_net.output, {self.state_ph:next_states_batch})  
            # Predict future Z distribution by passing next states and actions through value target network, also get target network's Z-atom values
            target_Z_dist, target_Z_atoms = self.sess.run([self.critic_target_net.output_probs, self.critic_target_net.z_atoms], {self.state_ph:next_states_batch, self.action_ph:future_action})
            # Create batch of target network's Z-atoms
            target_Z_atoms = np.repeat(np.expand_dims(target_Z_atoms, axis=0), train_params.BATCH_SIZE, axis=0)
            # Value of terminal states is 0 by definition
            target_Z_atoms[terminals_batch, :] = 0.0
            # Apply Bellman update to each atom
            target_Z_atoms = np.expand_dims(rewards_batch, axis=1) + (target_Z_atoms*np.expand_dims(gammas_batch, axis=1))
            # Train critic
            TD_error, _ = self.sess.run([self.critic_net.loss, self.critic_train_step], {self.state_ph:states_batch, self.action_ph:actions_batch, self.target_Z_ph:target_Z_dist, self.target_atoms_ph:target_Z_atoms, self.weights_ph:weights_batch})   
            # Use critic TD errors to update sample priorities
            self.PER_memory.update_priorities(idx_batch, (np.abs(TD_error)+train_params.PRIORITY_EPSILON))
                        
            # Actor training step
            # Get policy network's action outputs for selected states
            actor_actions = self.sess.run(self.actor_net.output, {self.state_ph:states_batch})
            # Compute gradients of critic's value output distribution wrt actions
            action_grads = self.sess.run(self.critic_net.action_grads, {self.state_ph:states_batch, self.action_ph:actor_actions})
            # Train actor
            self.sess.run(self.actor_train_step, {self.state_ph:states_batch, self.action_grads_ph:action_grads[0]})
            
            # Update target networks
            self.sess.run(self.update_op)
            
            # Increment beta value at end of every step   
            priority_beta += beta_increment
                            
            # Periodically check capacity of replay mem and remove samples (by FIFO process) above this capacity
            if train_step % train_params.REPLAY_MEM_REMOVE_STEP == 0:
                if len(self.PER_memory) > train_params.REPLAY_MEM_SIZE:
                    # Prevent agent from adding new experiences to replay memory while learner removes samples
                    self.run_agent_event.clear()
                    samples_to_remove = len(self.PER_memory) - train_params.REPLAY_MEM_SIZE
                    self.PER_memory.remove(samples_to_remove)
                    # Allow agent to continue adding experiences to replay memory
                    self.run_agent_event.set()
                    
            sys.stdout.write('\rStep {:d}/{:d}'.format(train_step, train_params.NUM_STEPS_TRAIN))
            sys.stdout.flush()  
            
            # Save ckpt periodically
            if train_step % train_params.SAVE_CKPT_STEP == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=train_step)
                sys.stdout.write('\nCheckpoint saved.\n')   
                sys.stdout.flush() 
        
        # Stop the agents
        self.stop_agent_event.set()                            
