import numpy as np
from optparse import OptionParser
from models.grid import Grid
from models.vi_and_pi import value_iteration
from renderer.renderer import GridRenderer
from models.vi_and_pi import policy_iteration
from matplotlib import pyplot as plt

# Use the OptionParser library to get command line arguments
# for us, such as the file we want to load in.
def get_options():
    parser = OptionParser(usage="usage: %prog -l filename",
                          version="%prog 1.0")

    parser.add_option("-f", "--file",
                      action="store", # optional because action defaults to "store"
                      dest="file",
                      help="file to load",)

    return parser.parse_args()

# Just a very simple script to render a single grid, for sanity checking visually.
def main():
  options, args = get_options()

  file = options.file 


  losses = list()
  try:
    limit = 10
    counter = 0
    avg = 0.0
    print("Open file...")
    with open(file +".txt", 'r') as filehandle:
      for line in filehandle:
        curr_loss = float(line[:-1])

        avg += curr_loss
        counter += 1

        if counter == limit:
          losses.append(avg / float(limit))
          avg = 0.01
          counter = 0

      print("got all losses: ", len(losses))


  except IOError:
    pass

  plt.ylim((0, 6))
  plt.xlim((0, len(losses)))
  plt.xlabel('Training Iterations (Average over every 10 steps)')
  plt.ylabel('Huber Loss')
  plt.title('5x5 Training')

  plt.plot(losses)
  plt.show()   

# Program entry point.
if __name__ == "__main__":
	main()
