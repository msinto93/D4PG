'''
## Test ##
# Test a trained D4PG network. This can be run alongside training by running 'test_every_new_ckpt.py'.
@author: Mark Sinton (msinto93@gmail.com) 
'''

import tensorflow as tf
import numpy as np

from params import test_params
from agent import Agent


def test():
    # Set random seeds for reproducability
    np.random.seed(test_params.RANDOM_SEED)
    tf.set_random_seed(test_params.RANDOM_SEED)
         
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)     
            
    # Initialise agent
    agent = Agent(sess, test_params.ENV, test_params.RANDOM_SEED)
    # Build network
    agent.build_network(training=False)
    
    # Test network
    agent.test()
    
    sess.close()
    
    
if  __name__ == '__main__':
    test()    
    
    
    
    
    
    
    