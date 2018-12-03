'''
## Train ##
# Code to train D4PG Network on OpenAI Gym environments
@author: Mark Sinton (msinto93@gmail.com) 
'''
import threading
import random
import tensorflow as tf
import numpy as np

from params import train_params
from utils.prioritised_experience_replay import PrioritizedReplayBuffer   
from utils.gaussian_noise import GaussianNoiseGenerator
from agent import Agent
from learner import Learner
    
          
def train():
    
    tf.reset_default_graph()
    
    # Set random seeds for reproducability
    np.random.seed(train_params.RANDOM_SEED)
    random.seed(train_params.RANDOM_SEED)
    tf.set_random_seed(train_params.RANDOM_SEED)
    
    # Initialise prioritised experience replay memory
    PER_memory = PrioritizedReplayBuffer(train_params.REPLAY_MEM_SIZE, train_params.PRIORITY_ALPHA)
    # Initialise Gaussian noise generator
    gaussian_noise = GaussianNoiseGenerator(train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.NOISE_SCALE)
            
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)  
    
    # Create threads for learner process and agent processes       
    threads = []
    # Create threading events for communication and synchronisation between the learner and agent threads
    run_agent_event = threading.Event()
    stop_agent_event = threading.Event()
    
    # with tf.device('/device:GPU:0'):
    # Initialise learner
    learner = Learner(sess, PER_memory, run_agent_event, stop_agent_event)
    # Build learner networks
    learner.build_network()
    # Build ops to update target networks
    learner.build_update_ops()
    # Initialise variables (either from ckpt file if given, or from random)
    learner.initialise_vars()
    # Get learner policy (actor) network params - agent needs these to copy latest policy params periodically
    learner_policy_params = learner.actor_net.network_params + learner.actor_net.bn_params
    
    threads.append(threading.Thread(target=learner.run))
    
    
    for n_agent in range(train_params.NUM_AGENTS):
        # Initialise agent
        agent = Agent(sess, train_params.ENV, train_params.RANDOM_SEED, n_agent)
        # Build network
        agent.build_network(training=True)
        # Build op to periodically update agent network params from learner network
        agent.build_update_op(learner_policy_params)
        # Create Tensorboard summaries to save episode rewards
        if train_params.LOG_DIR is not None:
            agent.build_summaries(train_params.LOG_DIR + ('/agent_%02d' % n_agent))
 
        threads.append(threading.Thread(target=agent.run, args=(PER_memory, gaussian_noise, run_agent_event, stop_agent_event)))
    
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
    
    sess.close()
    
if  __name__ == '__main__':
    train()         
            
        
    
    
        
        