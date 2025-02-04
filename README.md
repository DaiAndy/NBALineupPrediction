# NBALineupPrediction

Setup: 

1. nvidia smi - checks for cuda compatibility
2. install anaconda3 for windows - add path to system variables
3. install pycharm
4. new project - custom - virtualenv
5. install cuda toolkit if running cuda, if no cuda compatible devices:
      import os
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces PyTorch to use CPU

6. installing imports: 
      install torch
      install pandas
