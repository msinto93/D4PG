'''
## Test Every New Ckpt ##
# This allows testing to be run alongside training by running 'run_every_new_ckpt.sh', which monitors the ckpt directory and runs test.py every time a new ckpt is added.
@author: Mark Sinton (msinto93@gmail.com) 
'''

from subprocess import call
from params import test_params

if  __name__ == '__main__':

    ckpt_dir = test_params.CKPT_DIR
    call(['bash', 'utils/run_every_new_ckpt.sh', ckpt_dir])