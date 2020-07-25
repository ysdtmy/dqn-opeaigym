import os
import sys
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

import random
import unittest
from train import ExperienceReplayd


class TestExperienceReplayd(unittest.TestCase):

    def test_add_experice(self):
        memory_size = 2
        memory = ExperienceReplayd(memory_size)
        exp = (1,2,3,4,5)
        memory.add(exp)
        actual = memory.sample(1)

        self.assertEqual(exp, actual[0])    

    def test_pop(self):
        random.seed(71)
        exp_length = 100
        exp_first = (900, 4, 37, 73)
        exp_sample = [(936, 57, 32, 26)]

        memory = ExperienceReplayd(100)
        for i in range(1000):
            state = i
            action =random.randint(1,100) 
            reward = random.randint(1,100)
            nextstate = random.randint(1,100)
            memory.add((state, action,  reward, nextstate))
    
        actual_length = memory.get_cnt()
        actual_first = memory.get(0)
        actual_sample = memory.sample(1)

        self.assertEqual(exp_length, actual_length)    
        self.assertEqual(exp_first, actual_first)    
        self.assertEqual(exp_sample, actual_sample)    

if __name__ == "__main__":
    unittest.main()