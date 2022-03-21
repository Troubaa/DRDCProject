"""Represents the full set of functions.

PySC2 couldn't use namedtuple since python3 has a limit of 255 function arguments, so
build something similar.

Therefore for our simulator we will use a namedtuple.
"""

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define the static list of types and actions for SC2."""

import collections
import numbers

import six
from pysc2.lib import point



def no_op(action):
  del action

def select_point(action, select_point_act, screen):
  """Select a unit at a point."""
  select = action.action_feature_layer.unit_selection_point
  screen.assign_to(select.selection_screen_coord)
  select.type = select_point_act



class ArgumentType(collections.namedtuple(
    "ArgumentType", ["id", "name", "sizes", "fn"])):
  """Represents a single argument type.

  Attributes:
    id: The argument id. This is unique.
    name: The name of the argument, also unique.
    sizes: The max+1 of each of the dimensions this argument takes.
    fn: The function to convert the list of integers into something more
        meaningful to be set in the protos to send to the game.
  """
  __slots__ = ()

  def __str__(self):
    return "%s/%s %s" % (self.id, self.name, list(self.sizes))

  @classmethod
  def point(cls):  # No range because it's unknown at this time.
    """Create an ArgumentType that is represented by a point.Point."""
    return cls(-1, "<none>", (0, 0), lambda a: point.Point(*a).floor())



class Arguments(collections.namedtuple("Arguments", [
    "screen"])):
  """The full list of argument types.

  Take a look at TYPES and FUNCTION_TYPES for more details.

  Attributes:
    screen: A point on the screen.
  """
  ___slots__ = ()

  @classmethod
  def types(cls, **kwargs):
    """Create an Arguments of the possible Types."""
    named = {name: type_._replace(id=Arguments._fields.index(name), name=name)
             for name, type_ in six.iteritems(kwargs)}
    return cls(**named)


# The list of known types.
TYPES = Arguments.types(
    screen=ArgumentType.point(),
)

# Which argument types do each function need?
FUNCTION_TYPES = {
    no_op: [],
    select_point: [TYPES.screen],
}

always = lambda _: True

class Function(collections.namedtuple(
    "Function", ["id", "name", "ability_id", "general_id", "function_type",
                 "args", "avail_fn"])):
  """Represents a function action.

  Attributes:
    id: The function id, which is what the agent will use.
    name: The name of the function. Should be unique.
    ability_id: The ability id to pass to sc2.
    general_id: 0 for normal abilities, and the ability_id of another ability if
        it can be represented by a more general action.
    function_type: One of the functions in FUNCTION_TYPES for how to construct
        the sc2 action proto out of python types.
    args: A list of the types of args passed to function_type.
    avail_fn: For non-abilities, this function returns whether the function is
        valid.
  """
  __slots__ = ()

  @classmethod
  def ui_func(cls, id_, name, function_type, avail_fn=always):
    """Define a function representing a ui action."""
    return cls(id_, name, 0, 0, function_type, FUNCTION_TYPES[function_type], avail_fn)

  @classmethod
  def spec(cls, id_, name, args):
    """Create a Function to be used in ValidActions."""
    return cls(id_, name, None, None, None, args, None)

  def __hash__(self):  # So it can go in a set().
    return self.id

  def __str__(self):
    return self.str()

  def str(self, space=False):
    """String version. Set space=True to line them all up nicely."""
    return "%s/%s (%s)" % (str(self.id).rjust(space and 4),
                           self.name.ljust(space and 50),
                           "; ".join(str(a) for a in self.args))


class Functions(object):
  """Represents the full set of functions.

  Can't use namedtuple since python3 has a limit of 255 function arguments, so
  build something similar.
  """

  def __init__(self, functions):
    self._func_list = functions
    self._func_dict = {f.name: f for f in functions}
    if len(self._func_dict) != len(self._func_list):
      raise ValueError("Function names must be unique.")

  def __getattr__(self, name):
    return self._func_dict[name]

  def __getitem__(self, key):
    if isinstance(key, numbers.Number):
      return self._func_list[key]
    return self._func_dict[key]

  def __iter__(self):
    return iter(self._func_list)

  def __len__(self):
    return len(self._func_list)


# pylint: disable=line-too-long
FUNCTIONS = Functions([
    Function.ui_func(0, "no_op", no_op),
    Function.ui_func(1, "attack_target_1", select_point),
    Function.ui_func(2, "attack_target_2", select_point),
    Function.ui_func(3, "attack_target_3", select_point),
    Function.ui_func(4, "attack_target_4", select_point),
    Function.ui_func(5, "attack_target_5", select_point),
    Function.ui_func(6, "attack_target_6", select_point),
    Function.ui_func(7, "attack_target_7", select_point),
    Function.ui_func(8, "attack_target_8", select_point),
])
# pylint: enable=line-too-long

# Some indexes to support features.py and action conversion.
ABILITY_IDS = collections.defaultdict(set)  # {ability_id: {funcs}}
for func in FUNCTIONS:
  if func.ability_id >= 0:
    ABILITY_IDS[func.ability_id].add(func)
ABILITY_IDS = {k: frozenset(v) for k, v in six.iteritems(ABILITY_IDS)}
FUNCTIONS_AVAILABLE = {f.id: f for f in FUNCTIONS if f.avail_fn}


class FunctionCall(collections.namedtuple(
    "FunctionCall", ["function", "arguments"])):
  """Represents a function call action.

  Attributes:
    function_id: Store the function id, eg 2 for select_point.
    arguments: The list of arguments for that function, each being a list of
        ints. For select_point this could be: [[0], [23, 38]].
  """
  __slots__ = ()

  @classmethod
  def all_arguments(cls, function, arguments):
    """Helper function for creating `FunctionCall`s with `Arguments`.

    Args:
      function: The value to store for the action function.
      arguments: The values to store for the arguments of the action. Can either
        be an `Arguments` object, a `dict`, or an iterable. If a `dict` or an
        iterable is provided, the values will be unpacked into an `Arguments`
        object.

    Returns:
      A new `FunctionCall` instance.
    """
    if isinstance(arguments, dict):
      arguments = Arguments(**arguments)
    elif not isinstance(arguments, Arguments):
      arguments = Arguments(*arguments)
    return cls(function, arguments)


class ValidActions(collections.namedtuple(
    "ValidActions", ["types", "functions"])):
  """The set of types and functions that are valid for an agent to use.

  Attributes:
    types: A namedtuple of the types that the functions require. Unlike TYPES
        above, this includes the sizes for screen and minimap.
    functions: A namedtuple of all the functions.
  """
  __slots__ = ()
