'''
## Test ##
# Test a trained D4PG network. This can be run alongside training by running 'run_every_new_ckpt.sh'.
@author: Mark Sinton (msinto93@gmail.com) 
'''

import tensorflow as tf
import numpy as np

from params import play_params
from agent import Agent


def play():
    # Set random seeds for reproducability
    np.random.seed(play_params.RANDOM_SEED)
    tf.set_random_seed(play_params.RANDOM_SEED)
         
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)     
            
    # Initialise agent
    agent = Agent(sess, play_params.ENV, play_params.RANDOM_SEED)
    # Build network
    agent.build_network(training=False)
    
    # Run network in environment
    agent.play()
    
    sess.close()
    
    
if  __name__ == '__main__':
    play()    
    
    
    
    
    
    
    