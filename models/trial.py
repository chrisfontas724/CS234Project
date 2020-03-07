import numpy as np
from optparse import OptionParser
import itertools
import copy
import sys

# Maps the action index to a specific move. The action numbers
# 0,1,2,3 correspond to (up, right, down, left) - going around
# clockwise. The first coordinate is the row and the second
# coordinate is the column.
action_map = {
    0: (-1, 0),  # Up
    1: (0, 1),  # Right
    2: (1, 0),  # Down
    3: (0, -1)  # Left
}


# This class represents the board to be used in flow free. It has square dimensions N and
# contains a number of pairs of colored dots located at unique locations (i,j).
# TODO: Fill this out more.
class Grid:

    # Quick initialization that does not rely on reading files.
    @staticmethod
    def create(spaces, num_colors, start_coords, end_coords):
        grid = Grid("")
        grid.size = len(spaces[0])
        grid.spaces = spaces
        grid.num_cols = num_colors
        grid.color_start_coords = start_coords
        grid.end_coords = end_coords
        grid.reset()
        return grid

    def __init__(self, filename=""):
        if filename == "":
            self.size = 0
            self.spaces = np.zeros((0, 0))
            self.color_start_coords = dict()
            self.color_end_coords = dict()
            self.start_state = None
            return

        file = open(filename, "r")
        counter = 0
        while True:
            line = file.readline()
            if not line:
                break

            # Grab the dimensions of the grid from the
            # first line.
            if counter == 0:
                # Read the line, remove unneeded characters, convert to list then convert
                # each item in the list to an integer.
                dimensions = [int(numeric_string) for numeric_string in line.replace("\n", "").split(' ')]

                # We should have two entries, otherwise the file format is incorrect.
                assert len(dimensions) == 2

                # Initialize the size and the 2D numpy
                # array that will represent the board.
                self.size = dimensions[0]
                self.spaces = np.zeros((self.size, self.size))

            # Else populate the current row we are on from the data in the line.
            else:
                # Kind of hacky...but basically just removes the invalid characters I found and converts
                # them to integers if they're valid numbers.
                row = [int(numeric_string) for numeric_string in line.replace("\n", "").replace('\t', ' ').split(' ') if
                       numeric_string is not '']

                if len(row) == 0:
                    continue

                # Row length should match the size, otherwise the file format is wrong.
                assert len(row) == self.size, \
                    "length is %r but should be %r" % (len(row), self.size)

                # Assign row to numpy array.
                self.spaces[counter - 1] = row

            counter += 1

            # Count the number of colors.
            color_set = set()
            for x in range(self.size):
                for y in range(self.size):
                    if self.spaces[x][y] != 0:
                        color_set.add(self.spaces[x][y])
            self.num_cols = len(color_set)

        file.close()

        self.reset()

    class State:
        def __init__(self, info, spaces, tips):
            self.info = info
            self.spaces = spaces.copy()
            self.tips = tips.copy()

        def __eq__(self, other):
            return np.array_equal(self.spaces, other.spaces) and \
                   self.tips == other.tips

        def __ne__(self, other):
            return not self.__eq__(other)

        def __hash__(self):
            return hash((str(self.spaces.tolist()), str(self.tips)))

        # Checks to see if the provided action is allowed. Actions
        # are allowed if they do not cause a flow to go out of bounds
        # or if they do not result in a flow intersecting itself. Flows
        # are allowed to intersect flows of different colors.
        def is_viable_action(self, action, check_end_tips=True):
            if not action[0] in self.tips:
                raise Exception("Action color " + str(action[0]) + " not found in tips...")

            tip = self.tips[action[0]]
            direction = action_map[action[1]]
            new_pos = (tip[0] + direction[0], tip[1] + direction[1])

            # Make sure the new position is in bounds.
            if new_pos[0] < 0 or new_pos[0] >= self.info.size or \
                    new_pos[1] < 0 or new_pos[1] >= self.info.size:
                return False

            # If we're not checking end tips, we should make sure the space is
            # empty (0).
            if not check_end_tips:
                if self.spaces[new_pos[0]][new_pos[1]] != 0:
                    return False

            # Make sure the new position doesn't intersect a fixed
            # starting point.
            for _, value in self.info.color_start_coords.items():
                if new_pos == value:
                    return False

            # Make sure the new position doesn't intersect a fixed
            # ending point, unless its the ending point of its color.
            if check_end_tips:
                for key, value in self.info.color_end_coords.items():
                    if key != action[0] and new_pos == value:
                        return False

            # Make sure the new position doesn't already contain
            # the current flow *UNLESS* it's the end state, then
            # we can go into it (this is important for determining
            # if we are in the winning state or not).
            return (self.spaces[new_pos[0]][new_pos[1]] != action[0] or \
                    (check_end_tips and new_pos == self.info.color_end_coords[action[0]]))

        # Given an input state, return a list of possible actions that can be
        # taken from the provided state. At most, the number of moves is
        # (num_colors * num_directions).
        def possible_actions(self, check_end_tips=True):
            # The list of all action tuples you can take from the
            # current board configuration.
            result = list()

            # Loop over all possible colors:
            for col in range(1, self.info.num_cols + 1):
                # Loop over actions in the action map.
                for key, value in action_map.items():
                    test_action = (col, key)

                    if self.is_viable_action(test_action, check_end_tips):
                        result.append(test_action)

            return result

        # Given a state paired with a particular action, return the next
        # state that would be resulted in. Actions are defined as tuples
        # consisting of a color to move, and the direction in which to
        # move it. For example (red, up) and (blue, left). These values
        # are represented numerically, however, so (red, up) would be
        # (1,0) where "1" represents the color red and "0" represents
        # the direction up.
        def next_state(self, action_tuple, check_end_tips=True):

            # Just do a basic check at the beginning to make sure we're not
            # passing a bad action.
            if not self.is_viable_action(action_tuple, check_end_tips):
                raise Exception('Action is not viable')

            # Get the color and direction from the action.
            color, action = action_tuple

            # Copy the state, which will be returned at the end.
            result = copy.deepcopy(self)

            # We can  only move a color from its endpoint, so we have to find what
            # that endpoint is given the current state of the board. Then we can
            # append the action to it.
            tip = self.tips[color]

            # Update the resulting state based on the move.
            direction = action_map[action]
            new_tip = (tip[0] + direction[0], tip[1] + direction[1])

            # If the action moves the flow into a space that is already occupied
            # then that means it has interrupted another flow, so that other flow
            # must be reset.
            def break_flow():
                existing_value = self.spaces[new_tip[0]][new_tip[1]]
                if existing_value != 0 and existing_value != color:
                    for row in range(self.info.size):
                        for col in range(self.info.size):
                            item = int(self.spaces[row][col])

                            # Reset the space to zero if the color matches that of the broken flow and
                            # if it is neither a start point nor an end point.
                            if item == existing_value and \
                                    self.info.color_start_coords[item] != (row, col) and \
                                    (check_end_tips and self.info.color_end_coords[item] != (row, col)):
                                result.spaces[row][col] = 0
                                result.tips[existing_value] = result.info.color_start_coords[existing_value]

            # Call break flow.
            break_flow()

            # Update the board.
            result.spaces[new_tip[0]][new_tip[1]] = color
            result.tips[color] = new_tip

            # return the state.
            return result

        # Checks to see if the provided state is a winning state or not. To be
        # a winning state, all spaces must be covered and all starting and end
        # flows must be connected for every color.
        def is_winning(self, check_end_tips=True):
            # Make sure the are no 0s on the board.
            for x in range(self.info.size):
                for y in range(self.info.size):
                    if self.spaces[x][y] == 0:
                        return False

            # In most cases, such as when running actual RL algorithms we want
            # to check the tips, but for certain state-generation tasks we do
            # not care about this.
            if check_end_tips:
                # Finally, make sure all the flows have reached their end goal.
                for color in range(1, self.info.num_cols + 1):
                    if self.tips[color] != self.info.color_end_coords[color]:
                        return False

            # Make sure we have at least 2 of every color.
            else:
                counts = dict()
                for x in range(self.info.size):
                    for y in range(self.info.size):
                        col = self.spaces[x][y]
                        if col == 0:
                            continue
                        if col not in counts:
                            counts[col] = 1
                        else:
                            counts[col] += 1
                for color in range(1, self.info.num_cols + 1):
                    if counts[color] < 2:
                        return False

            # We win!
            return True

        def set_to_start_with_current_tips(self):
            self.info.color_end_coords = self.tips
            for x in range(self.info.size):
                for y in range(self.info.size):
                    color = self.spaces[x][y]
                    if self.info.color_start_coords[color] != (x, y) and \
                            self.info.color_end_coords[color] != (x, y):
                        self.spaces[x][y] = 0

                        # Generate all possible states by iterating over all possible actions for every

    # state, and then placing the newly generated state onto the stack so the process
    # can be repeated until all states have been covered.
    def generate_all_states(self):
        result = set()
        seen = set()
        stack = [self.start_state]

        while stack:
            curr = stack.pop()
            result.add(curr)
            possible_actions = curr.possible_actions()
            for action in possible_actions:
                next_state = curr.next_state(action)
                if next_state not in seen:
                    stack.append(next_state)
                    seen.add(next_state)

        return result

    # Resets the grid to the starting state, before any moves have been made.
    # Declare dictionaries to store the starting and ending positions for the
    # "flows" for each color. For simplicity, and due to the fact that it doesn't
    # affect the algorithm at all, we assume that all flows must start from one
    # color and make its way to the other, instead of allowing the flow to start
    # from either color arbitrarily. We also keep track of the current tip of the
    # flow to keep calculating states easier.
    def reset(self):
        self.color_start_coords = dict()
        self.color_end_coords = dict()
        self.color_flow_tips = dict()
        for row in range(self.size):
            for col in range(self.size):
                item = int(self.spaces[row][col])
                if item != 0:
                    if item in self.color_start_coords:
                        assert item not in self.color_end_coords
                        self.color_end_coords[item] = (row, col)
                    else:
                        self.color_start_coords[item] = (row, col)
        self.start_state = Grid.State(self, self.spaces, self.color_start_coords)
