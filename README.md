# Mirage - Friendly Node Interfaces for Maya

`Mirage` is a convenient interface to Maya's boilerplate-heavy OpenMaya Api 2.0.

The test suite runs against Maya 2024, but any version of maya above 2018 should
work!

Full API documentation will be coming soon!

Overview
--------
Maya nodes are represented by `MirageNode`s, and attributes by `MirageAttr`s.

Example usage:

```python
# move all mesh transforms by 3 units on X and Z
mesh_nodes = mirage.ls(type="mesh")
xforms = [n.parent for n in mesh_nodes]
for xform in xforms:
    xform["transform"] += [3, 0, 3]
```

```python
# Make a rainbow ramp node and apply it to a shader
sphere = MirageNode("mySphere")
ramp = mirage.shading_node("ramp", asTexture=True)
ramp["colorEntryList"] = [
    [0.0, [1, 0, 0]],
    [0.5, [0, 1, 0]],
    [1.0, [0, 0, 1]],
]
surface = sphere.shading_group["surfaceShader"]
ramp["outColor"].connect_to(surface["outColor"])
```

```python
# Get and set node properties and hierarchies

# property aliases to node information
sphere = MirageNode("mySphere")
sphere.type_name    # "transform"
sphere.name         # shortest unique name
sphere.long_name    # long name including group
sphere.uuid         # unique identifier for the node
sphere.parent       # the parent of this node
sphere.children     # any direct children of this node
# ... etc.

# declarative interface
sphere.name = "foo"           # name is now "foo"
sphere.namespace = "FOO"      # "FOO:foo"
sphere.parent = another_node  # move under a parent
sphere.parent = None          # move back under the root
sphere.selected = True        # select the node
# ... etc.
```

The major design goals: *first, convenience*, *second, speed*.

Many of the properties and methods on the MirageNode are just thin wrappers to
OpenMaya's functionality with nicer names and interfaces; Others employ tricks
to evaluate the scene in an optimized way that is non-obvious or difficult to
employ on-the-fly when writing OpenMaya code.

Mirage is designed to provide a simple interface to many common
maya tasks, but its coverage of OpenMaya and cmds is **not comprehensive**.  In particular, specific support for animation data is lacking - there is currently no way to obtain an attribute's value at a specific time, for instance.

In later iterations, more features may be added.

Existing Solutions
------------------

This is not the first time we've re-invented this particular wheel.  `PyMel` has shipped with Maya for years (until v2024), but it's incredibly slow, and it is intrinsically tied to the semantics of the `cmds` module.  [Mottosso](https://github.com/mottosso)'s excellent [cmdx project](https://github.com/mottosso/cmdx) is a lot closer to Mirage in functionality, and is a great choice as a PyMel alternative as well.

Creating MirageNodes
--------------------

MirageNodes are obtained by default using the name of the node as a string.

Also by default, two MirageNode instances created from the same maya name, uuid,
or MObject will be two different Python objects:
```python
a = MirageNode.from_name("persp")
b = MirageNode.from_name("persp")
assert a is not b
```

If getting the same python object is important to your use-case, you can
use the from_*_cached versions of the constructors, which will always
return the same object at a small penalty in speed of instantiation::
```python
a = MirageNode.from_name_cached("persp")
b = MirageNode.from_name_cached("persp")
assert a is b
```

MirageNodes can also be obtained using uuids:
```python
a = MirageNode.from_uuid(...)
# or
a = MirageNode.from_uuid_cached(...)
```

any two MirageNodes built from the same underlying maya object share the same
hash, and can be used both as dictionary keys and compared for equivalency,
even if they aren't the same object in memory:

```python
a = MirageNode.from_name("persp")
b = MirageNode.from_name("persp")
node_dict = {a: True}

assert node_dict[b] is True
assert a == b
assert a is not b
```


Using Maya's `cmds` module
--------------------------
MirageNodes can also be generated by calling cmds module commands that return node
name strings.  Just pass the command to the "from_cmd" constructor and any
required keyword arguments.

for example, to get the current selection list as MirageNodes:
```python
nodes = MirageNode.from_cmd("ls", sl=True)
```
this is equivalent to the following comprehension:
```python
nodes = [MirageNode(name) for name in cmds.ls(sl=True)]
```
Note that **MirageNode.from_cmd will ALWAYS return a list of nodes**, even if the
list only has a single item.

Accessing Attributes
--------------------
Attributes on MirageNodes can be accessed through their names, using either
the "attr" method, or dictionary-style lookup (__getitem__).  Both
approaches return a MirageAttr object

```python
attr_one = my_mirage_node.attr("someAttribute")
attr_two = my_mirage_node["someAttribute"]
assert attr_one is attr_two
```

Setting Attribute Values:
-------------------------
MirageNodes provide a shortcut for setting attribute values.  You can
directly assign an attribute values dictionary-style (__setitem__):

```python
my_mirage_node["someAttribute"] = 5.0
```

This is equivalent to the long way:
```python
my_mirage_node.attr("someAttribute").value = 5.0
```
Adding New Dynamic Attributes:
------------------------------
MirageNode provides a simple interface for creating the same attribute
types as provided in the Maya UI, namely integer, floating-point,
boolean, color, string, and vector attributes.

The only required argument is the long name for the new attribute:

```python
my_mirage_node.add_string_attr("myNewAttr", short_name="newattr")
```

If a short name is not provided, one will be generated:

```python
new_attr = my_mirage_node.add_string_attr("myNewAttr")
assert new_attr.short_name == "mna"
```

These methods also provide two ways to set values on the newly-created
attributes, a `default_value`, and an `initial_value`.

It is valid and encouraged to use both.  The "default_value" sets a default
that is stored with the maya node and can be compared against, whereas
"initial_value" sets the value after creating the attribute.

for example:
```python
new_attr = my_mirage_node.add_string_attr(
    "myNewAttr",
    default_value="default",
    initial_value="initial"
)

assert new_attr.value == "initial"
assert not new_attr.is_default
```

Other ways to get MirageAttrs
-----------------------------

By default, MirageAttrs are instantiated using a maya MPlug object.

Also, alternate constructors are provided to instantiate MirageAttrs:

Using the full name of an attribute:
```python
attr = MirageAttr.from_full_name("my_node.tx")
```
Using an existing MirageNode and an attribute name:
```python
attr = MirageAttr.from_mirage_node_and_name(my_mirage_node, "tx")
```
