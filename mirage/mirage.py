import enum
import re
from functools import lru_cache
from contextlib import contextmanager

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TypeVar,
    Sequence,
)

from maya.api import OpenMaya
from maya import cmds

__all__ = [
    "MirageNode",
    "MirageAttr",
    "ConnectionTypes",
    "NodeName",
    "CommandString",
    "m_obj_from_name",
    "m_objs_from_names",
    "m_obj_from_uuid",
    "dg_modifier",
    "dag_modifier",
    "camel_case_split",
    "split_long_name",
    "get_short_attr_name",
    "mel_attr_fmt",
    "new_uuid",
    "get_world_node",
]

# aliases for better type annotation
CommandString = str
NodeName = str


# CACHED FUNCTION SETS
# These cached function set instances are re-used as needed by MirageNodes.
# This is a major optimization.


@lru_cache()
def _fn_dg() -> OpenMaya.MFnDependencyNode:
    return OpenMaya.MFnDependencyNode()


@lru_cache()
def _fn_dag() -> OpenMaya.MFnDagNode:
    return OpenMaya.MFnDagNode()


@lru_cache()
def _fn_set() -> OpenMaya.MFnSet:
    return OpenMaya.MFnSet()


@lru_cache()
def _fn_numeric_attr() -> OpenMaya.MFnNumericAttribute:
    return OpenMaya.MFnNumericAttribute()


@lru_cache()
def _fn_enum_attr() -> OpenMaya.MFnEnumAttribute:
    return OpenMaya.MFnEnumAttribute()


@lru_cache()
def _fn_typed_attr() -> OpenMaya.MFnTypedAttribute:
    return OpenMaya.MFnTypedAttribute()


@lru_cache()
def _fn_unit_attr() -> OpenMaya.MFnUnitAttribute:
    return OpenMaya.MFnUnitAttribute()


@lru_cache()
def _fn_attr() -> OpenMaya.MFnAttribute:
    return OpenMaya.MFnAttribute()


@lru_cache()
def _fn_transform() -> OpenMaya.MFnTransform:
    return OpenMaya.MFnTransform()


# CACHED MAPPINGS


@lru_cache()
def _fn_type_dict() -> Dict[int, str]:
    return _om_enum_to_dict(OpenMaya.MFn)


@lru_cache()
def _data_type_dict() -> Dict[int, str]:
    return _om_enum_to_dict(OpenMaya.MFnData)


@lru_cache()
def _numeric_type_dict() -> Dict[int, str]:
    return _om_enum_to_dict(OpenMaya.MFnNumericData)


@lru_cache()
def _unit_type_dict() -> Dict[int, str]:
    return _om_enum_to_dict(OpenMaya.MFnUnitAttribute)


# ENUMS


class ConnectionTypes(enum.Enum):
    ALL = 0
    SOURCE = 1
    DESTINATION = 2


# CLASSES


class MirageNode:
    """Wrappers around node functionality in OpenMaya Python Api 2."""

    #: A cache of MirageNode instances, keyed by their UUID
    _instances = dict()

    #: A cache of function sets for easy lookup
    _dag_fn_sets = dict()

    def __init__(self, maya_node_name: NodeName = ""):
        """
        Args:
            maya_node_name (str): The string representation of a node in maya.
                This name must be unique -- either a unique short name or a
                pipe-separated long name in the case of DAG nodes.  If no name
                is provided, the mirage node can remain empty until the node
                is set.

        Example::
            my_shader_node = MirageNode("aShaderNode1SG")
            my_dag_node = MirageNode("aDagNode1")
            my_long_name_node = MirageNode("grp1|grp2|aDagNode1")
        """

        # A MirageNode can be instantiated "empty" and populated on-demand.
        self._m_obj: OpenMaya.MObject
        if maya_node_name:
            self._m_obj = m_obj_from_name(maya_node_name)

        # defaults
        self._hash_code = None

        # attribute and plug caches
        self._plugs = {}
        self._attrs = {}
        self._dict_caches = [self._plugs, self._attrs]

        # other cached values
        self._cached_m_dag_path = None

    # CONSTRUCTORS

    @staticmethod
    def _from_cmd_generic(
        factory, cmd: CommandString, *args, **kwargs
    ) -> Union["MirageNode", List["MirageNode"]]:
        """Uses MirageNode.from_cmd to generate MirageNode instances."""
        node_or_nodes = getattr(cmds, cmd)(*args, **kwargs)

        # All maya cmds commands that return more will return a list, this is
        # a safe assumption to make.
        if isinstance(node_or_nodes, list):
            return [factory(x) for x in node_or_nodes]
        return factory(node_or_nodes)

    @classmethod
    def from_cmd(cls, cmd: CommandString, *args, **kwargs) -> List["MirageNode"]:
        """Alt Constructor: Builds a new MirageNode from a maya command.

        Args:
            cmd (str): The name of a maya command found in maya.cmds
            *args: Any positional arguments for the maya command
            **kwargs: Any keyword-arguments for the maya command

        Returns:
            MirageNode or list: If the underlying command returns more
                than one node (i.e. a transform and a shape), a list with all
                the maya nodes as MirageNodes will be returned.

        Example::
            # get all maya transform nodes as MirageNodes:
            all_nodes = MirageNode.from_cmd("ls", type="transform")
        """
        ret = cls._from_cmd_generic(cls.from_name, cmd, *args, **kwargs)
        if not isinstance(ret, list):
            return [ret]
        return ret

    @classmethod
    def from_name(cls, maya_node_name: NodeName) -> "MirageNode":
        """Alt Constructor: Returns the MirageNode for a maya node name.

        Args:
            maya_node_name (str): A valid maya node name.
        """
        return cls(maya_node_name)

    @classmethod
    def from_name_cached(cls, maya_node_name: NodeName) -> "MirageNode":
        """Alt Constructor: Returns a cached MirageNode for the node name.

        Args:
            maya_node_name (str): A valid maya node name.
        """
        return cls.from_cache(maya_node_name=maya_node_name)

    @classmethod
    def from_names(cls, maya_node_names: Iterable[NodeName]) -> List["MirageNode"]:
        """Alt Constructor: Returns a list of MirageNodes from node names.

        Args:
            maya_node_names (iterable): Any collection of valid node names.
        """
        return [cls.from_m_obj(m_obj) for m_obj in m_objs_from_names(maya_node_names)]

    @classmethod
    def from_names_cached(cls, maya_node_names: List[NodeName]) -> List["MirageNode"]:
        """Alt Constructor: Returns cached MirageNodes from maya node names.

        Args:
            maya_node_names (iterable): Any collection of valid node names.
        """
        return [
            cls.from_m_obj_cached(m_obj) for m_obj in m_objs_from_names(maya_node_names)
        ]

    @classmethod
    def create_node(cls, node_type: str, *args, **kwargs) -> "MirageNode":
        """Alt Constructor: create and return a node of the given type.

        Args:
            node_type (str): Any valid maya node type
            *args: positional arguments for creating the maya node using
                `cmds.createNode`
            **kwargs: keyword-arguments for creating the maya node using
                `cmds.createNode`
        """
        return cls.from_cmd("createNode", node_type, *args, **kwargs)  # type: ignore

    @classmethod
    def get_or_create_node(
        cls, name: NodeName, node_type: str, *args, **kwargs
    ) -> "MirageNode":
        """Alt Constructor: Gets or creates a MirageNode for the given name.

        If the node already exists, this function will return the MirageNode
        for it. If the node does not already exist, one will be created using
        `cmds.createNode`.

        Args:
            name (str): The maya node name
            node_type (str): A valid maya or plugin node type
        """
        if cmds.objExists(name):
            return cls.from_name(name)
        return cls.create_node(node_type, name=name, *args, **kwargs)

    @classmethod
    def get_or_create_node_cached(
        cls, name: NodeName, node_type: str, *args, **kwargs
    ) -> "MirageNode":
        """Alt Constructor: Gets or creates a MirageNode for the given name.

        This method will return the cached version of the MirageNode if it
        exists for the given name.  If the node does not already exist, one
        will be created using `cmds.createNode`.

        Args:
            name (str): The maya node name
            node_type (str): A valid maya or plugin node type
        """
        if cmds.objExists(name):
            return cls.from_name_cached(name)
        return cls.create_node(node_type, name=name)

    @classmethod
    def from_uuid(cls, uuid_as_string: str) -> "MirageNode":
        """Alt Constructor: Returns a MirageNode for the given UUID.

        Args:
            uuid_as_string (str): The maya node UUID.
        """
        instance = cls.from_m_obj(m_obj_from_uuid(uuid_as_string))
        instance._uuid = uuid_as_string
        return instance

    @classmethod
    def from_uuid_cached(cls, uuid_as_string: str) -> "MirageNode":
        """Returns the cached version of the MirageNode for the given UUID.

        If the MirageNode does not exist, caches and returns the results.

        Args:
            uuid_as_string (str): The maya node UUID.
        """
        return cls.from_cache(uuid_as_str=uuid_as_string)

    @classmethod
    def from_m_obj(cls, m_obj: OpenMaya.MObject) -> "MirageNode":
        """Alt Constructor: Returns a MirageNode for the given MObject.

        Args:
            m_obj (MObject): A maya node object from OpenMaya 2.
        """
        instance = cls()
        instance._m_obj = m_obj
        return instance

    @classmethod
    def from_m_obj_cached(cls, m_obj: OpenMaya.MObject) -> "MirageNode":
        """Returns the cached version of a MirageNode for the MObject, if any.

        Args:
            m_obj (MObject): A maya node object from OpenMaya 2.
        """
        return cls.from_cache(m_obj=m_obj)

    @classmethod
    def from_m_objs(cls, m_objs: Iterable[OpenMaya.MObject]) -> List["MirageNode"]:
        """Returns a list of MirageNodes for an iterable of MObjects."""
        return [cls.from_m_obj(m_obj) for m_obj in m_objs]

    @classmethod
    def from_m_objs_cached(
        cls, m_objs: Iterable[OpenMaya.MObject]
    ) -> List["MirageNode"]:
        """Returns the cached MirageNodes for an iterable of MObjects."""
        return [cls.from_m_obj_cached(m_obj) for m_obj in m_objs]

    @classmethod
    def from_cache(
        cls,
        maya_node_name: Optional[NodeName] = None,
        uuid_as_str: Optional[str] = None,
        m_obj: Optional[OpenMaya.MObject] = None,
    ) -> "MirageNode":
        """Returns a cached MirageNode from a name, uuid, or MObject."""

        if not any((maya_node_name, uuid_as_str, m_obj)):
            return cls()

        if maya_node_name is not None:
            m_obj = m_obj_from_name(maya_node_name)

        if uuid_as_str is not None:
            m_obj = m_obj_from_uuid(uuid_as_str)

        handle = OpenMaya.MObjectHandle(m_obj)
        hash_code = handle.hashCode()

        mirage_node = cls._instances.setdefault(hash_code, cls())
        mirage_node._hash_code = hash_code
        mirage_node._m_obj = m_obj

        return mirage_node

    # NODE FUNCTION SETS

    @property
    def fn_dg(self) -> OpenMaya.MFnDependencyNode:
        """Returns a DG function set object for this MirageNode."""
        return _fn_dg_for_m_obj(self._m_obj)

    @property
    def fn_dag(self) -> OpenMaya.MFnDagNode:
        """Returns a DAG function set object for this MirageNode."""
        return _fn_dag_for_m_obj(self._m_obj)

    @property
    def fn_set(self) -> OpenMaya.MFnSet:
        """Returns a MFnSet object for this MirageNode.

        If this property is called from a non-set-like node, a RuntimeError
        will be raised by Maya.
        """
        return _fn_set_for_m_obj(self._m_obj)

    @property
    def fn_transform(self) -> OpenMaya.MFnTransform:
        """Returns a MFnTransform object for this MirageNode.

        If this property is called from a non-transform node, a Runtime error
        will be raised.
        """
        return _fn_transform_for_m_obj(self._m_obj)

    # OBJECT ID PROPERTIES

    @property
    def _handle(self) -> OpenMaya.MObjectHandle:
        """The MObjectHandle for this MirageNode.

        A maya "handle" tracks an object during its runtime lifecycle, a bit
        like a pointer.  The handle can be used to track a node after it has
        been deleted from a scene, but not between sessions.
        """
        return OpenMaya.MObjectHandle(self._m_obj)

    @property
    def hash_code(self) -> str:
        """The hash code for this MirageNode.

        The hash code is Maya's way of internally tracking an object.  It's
        a bit like a UUID that is only valid for a single session.  It is a
        serialized equivalent to the maya handle object.
        """
        if self._hash_code is None:
            self._hash_code = self._handle.hashCode()
        return self._hash_code

    @property
    def uuid(self) -> str:
        """The UUID for this MirageNode as a string.

        UUIDs are identifiers single nodes across multiple maya sessions and
        multiple file iterations or copies.
        """
        return self.fn_dg.uuid().asString()

    @uuid.setter
    def uuid(self, uuid_as_string: str):
        """Assigns a uuid string to this node's uuid attribute."""
        muuid = OpenMaya.MUuid(uuid_as_string)
        self._uuid = uuid_as_string
        self.fn_dg.setUuid(muuid)

    # NAME PROPERTIES

    @property
    def name(self) -> NodeName:
        """Returns the name of the node.

        If the node is a DAG object and has no other nodes with the same
        short name, the short name will be returned. If the short name is
        shared with another node, the unique long name will be returned. You
        can elect to only get the long name of DAG nodes by using the
        long_name property instead.
        """
        return self.fn_dg.name()

    @name.setter
    def name(self, new_name: NodeName):
        """Assigns a new name to this node.

        Note that this change requires a new modifier stack to be instantiated
        and run, so this operation can be a little expensive.

        Whenever possible, it's faster to provide a final node name on node
        creation or use a custom modifier stack to rename multiple objects at
        once.
        """
        with dg_modifier() as modifier:
            modifier.renameNode(self._m_obj, new_name)

    @property
    def short_name(self) -> NodeName:
        """The short name of this node.

        If the node is a DAG object, the name is not guaranteed to be unique.
        """
        return split_long_name(self.long_name)[-1]

    @property
    def long_name(self) -> NodeName:
        """The full path name of a DAG node, or the dependency node name."""
        try:
            return self.fn_dag.fullPathName()
        except RuntimeError:
            return self.name

    @property
    def namespace(self) -> str:
        """Returns the namespace this node belongs to."""
        return self.fn_dg.namespace

    @namespace.setter
    def namespace(self, namespace):
        """Sets the namespace of this node."""
        if namespace is None or namespace == "":
            namespace = ":"
        if not cmds.namespace(exists=namespace):
            cmds.namespace(addNamespace=namespace)
        full_name_split = split_long_name(self.long_name)
        name_without_namespace = full_name_split[-1].split(":")[-1]
        new_name = f"{namespace}:{name_without_namespace}"
        self.name = new_name

    # TYPING AND CLASSIFICATION

    @property
    def type_name(self) -> str:
        """The type name of the node.

        This is distinct from the api type, as the type name is a plain
        English name like "transform" or "mesh" rather than the api type,
        which will use the name of the type as used in Maya's C++ codebase
        (usually prefixed with "k" for constant)

        type_name -> "transform"
        api_type -> "kTransform"

        """
        return self.fn_dg.typeName

    @property
    def api_type(self) -> str:
        """Get the api type of the node.

        This is distinct from the type name, as the type name is a plain
        English name like "transform" or "mesh" rather than the api type,
        which will use the name of the type as used in Maya's C++ codebase
        (usually prefixed with "k" for constant)

        type_name -> "transform"
        api_type -> "kTransform"

        """
        assert self._m_obj
        return self._m_obj.apiTypeStr

    @property
    def classification(self) -> str:
        """The classification for a node as a string.

        Classifications are a bit like categories, as opposed to api_type or
        type_name, which are node-specific. Different node types can share a
        classification. Classifications are separated using forward slashes.
        """
        return OpenMaya.MFnDependencyNode.classification(self.type_name)

    @property
    def classifications(self) -> List[str]:
        """A list of this node's classifications as individual strings"""
        return self.classification.split("/")

    @property
    def inherited_types(self) -> List[str]:
        """A list of inherited node type names"""
        try:
            # here we use the cmds module to get the inherited types.  It's
            # almost as fast as OpenMaya's equivalent and saves a ton of messy
            # required boilerplate.
            return cmds.nodeType(self.name, inherited=True)

        # Maya likes to throw an error rather than return None for nodes that
        # do not inherit any types, like the WorldNode.
        except RuntimeError:
            return [self.type_name]

    # STATUS PROPERTIES

    @property
    def is_dag(self) -> bool:
        """True if the node is part of the DAG, False if not."""
        assert self._m_obj
        return self._m_obj.hasFn(OpenMaya.MFn.kDagNode)

    @property
    def alive(self) -> bool:
        """Whether the current node is "alive".

        An node is "alive" if it still exists anywhere in the maya session,
        including the undo queue.  A node can be deleted and still be alive.
        """
        return self._handle.isAlive()

    @property
    def valid(self) -> bool:
        """Whether the current node is "valid".

        Valid nodes are active in the current scene (not deleted).
        """
        return self._handle.isValid()

    @property
    def selected(self) -> bool:
        """Whether the current node is in the active user selection."""
        # this functionality has not yet been implemented in OpenMaya API 2, so
        # we fall back to the (much) slower cmds version here
        return self.name in cmds.ls(sl=True)

    @selected.setter
    def selected(self, value):
        """Add or remove the node from the active user selection list."""
        if value:
            cmds.select(self.name, replace=False)
        else:
            cmds.select(self.name, deselect=True)

    @property
    def is_default(self) -> bool:
        """Returns True if the given node is a "default" node.

        Default nodes exist in all maya scenes by default.
        """
        return self.fn_dg.isDefaultNode

    # LOCKING

    @property
    def locked(self) -> bool:
        """True if the node is locked."""
        return self.fn_dg.isLocked

    @locked.setter
    def locked(self, lock_status: bool):
        """Locks or unlocks the node.

        Args:
            lock_status (bool): Whether to lock or unlock the node.
        """
        self.fn_dg.isLocked = lock_status

    def lock(self):
        """Locks the node."""
        self.locked = True

    def unlock(self):
        """Unlocks the node."""
        self.locked = False

    # HIERARCHY

    @property
    def parent(self) -> Optional["MirageNode"]:
        """The direct parent of this node.

        Note that this method does not support instances, which are single
        objects with multiple parents.
        """
        return MirageNode.from_m_obj(self.fn_dag.parent(0))

    @property
    def parents(self) -> List["MirageNode"]:
        """The direct parents of this node.

        This method is only really useful when querying instances, which might
        have multiple parents.
        """
        fn_dag = self.fn_dag
        return [
            MirageNode.from_m_obj(fn_dag.parent(i)) for i in range(fn_dag.parentCount())
        ]

    @parent.setter
    def parent(self, other_node: Optional["MirageNode"]):
        """Sets the parent DAG node."""
        with dag_modifier() as modify_stack:
            if other_node is None:
                modify_stack.reparentNode(self._m_obj)
            else:
                modify_stack.reparentNode(self._m_obj, other_node._m_obj)

    def adopt(self, other_node: "MirageNode"):
        """Move node under this one, removing it from its previous parent."""
        other_node.parent = self

    @property
    def children(self) -> List["MirageNode"]:
        """All the direct children of this node."""
        # return MirageNode.from_m_objs(_m_obj_children(self._m_obj))
        children = []
        fn_dag = self.fn_dag
        for i in range(fn_dag.childCount()):
            children.append(fn_dag.child(i))
        return [MirageNode.from_m_obj(child) for child in children]

    @property
    def descendants(self) -> List["MirageNode"]:
        """All the descendants of this node.

        This includes any node that can be found by searching the children of
        this node, then their children, etc."""
        return MirageNode.from_m_objs(_m_obj_descendants(self._m_obj))

    @property
    def _m_dag_path(self) -> OpenMaya.MDagPath:
        """The MDagPath object for this node."""
        try:
            return self.fn_dag.getPath()
        except RuntimeError:
            # if the node is not DAG, then this will fire
            raise TypeError(
                "Error while trying to get the dag path for {node}. "
                "The node might not be a DAG object, or might not exist"
                "".format(node=self.name)
            )

    @property
    def shape(self) -> "MirageNode":
        """The shape node of a transform as a MirageNode, if any exists.

        If this node is already a shape node, it will be returned.

        Accessing this property from a non-DAG node will raise a TypeError.
        """
        try:
            m_dag_path = self._m_dag_path
        except TypeError:
            raise TypeError(
                "Error caught while trying to get the Shape of {}. The node "
                "might not be a DAG object, or might not have a Shape node "
                "at all".format(self.name)
            )

        try:
            m_dag_path.extendToShape()
        except Exception:
            raise TypeError(
                "Error caught while trying to get the Shape of {}. The node "
                "might not be a DAG object, or might not have a Shape node "
                "at all".format(self.name)
            )

        m_obj = m_dag_path.node()
        if m_obj == self._m_obj:
            return self
        else:
            return MirageNode.from_m_obj(m_obj)

    @property
    def ancestors(self) -> List["MirageNode"]:
        """A list of the ancestors of this node.

        This includes any node that can be found by searching the parents of
        this node, then their parents and their parents, etc."""
        return [MirageNode.from_m_obj(n) for n in _m_obj_ancestors(self._m_obj)]

    # NODE DELETION

    def delete(self):
        """Deletes this MirageNode's maya object from the scene."""
        # The cmds module is the safest and most-consistent way to properly
        # delete a node while maintaining the undo-queue as users expect
        cmds.delete(self.long_name)

    # ATTRIBUTE LOOKUP

    def plug(self, plug_name: str) -> OpenMaya.MPlug:
        """Retrieves the maya MPlug object for the given name.

        If the plug does not exist, an AttributeError is raised.

        Args:
            plug_name (str): The name of an attribute on the node.
        """
        try:
            return self._plugs[plug_name]
        except KeyError:
            try:
                plug = OpenMaya.MPlug(self.fn_dg.findPlug(plug_name, False))
            except RuntimeError:
                # Translate maya's RuntimeError into a formatted AttributeError
                raise AttributeError(
                    "the given node {node_name} does not have an attribute "
                    "called {plug_name}"
                    "".format(node_name=self.name, plug_name=plug_name)
                )

            return self._plugs.setdefault(plug_name, plug)

    def attr(self, attr_name: str, extend_to_shape: bool = True) -> "MirageAttr":
        """Gets the MirageAttr for the given attribute name.

        Args:
            attr_name (str): The name of the attribute to get a MirageAttr for

            extend_to_shape (bool): If True, then an attribute queried on a
                transform node will return the attr if it lives in the
                transform's Shape node. This is the default behavior in Maya's
                own getAttr function.
        """
        try:
            return MirageAttr.from_mirage_node_and_name(self, attr_name)
        except AttributeError as e:
            if not extend_to_shape:
                raise e
            try:
                assert self.shape
                return MirageAttr.from_mirage_node_and_name(self.shape, attr_name)
            except TypeError:
                raise e

    @property
    def all_plugs(self) -> List[OpenMaya.MPlug]:
        """List of all top-level plugs on this MirageNode

        Returns: List of MPlugs
        """
        fn_dg = self.fn_dg
        attr_objs = [fn_dg.attribute(i) for i in range(fn_dg.attributeCount())]
        plugs = [OpenMaya.MPlug(self._m_obj, attr) for attr in attr_objs]
        plugs = [plug for plug in plugs if not plug.isChild]
        self._plugs = {plug.partialName(useLongNames=True): plug for plug in plugs}
        return plugs

    @property
    def all_connected_plugs(self) -> List[OpenMaya.MPlug]:
        """Lists all MPlug objects with external connections."""
        try:
            plug_array = self.fn_dg.getConnections()

        # maya throws an error if no connections are found.  This has the
        # potential to mask other RuntimeErrors
        except RuntimeError:
            return []

        else:
            num_plugs = len(plug_array)

            # minor optimization, we already know how many plugs exist so we
            # can pre-allocate the list rather than appending.  This adds up.
            plugs = [Any] * num_plugs

            for i in range(num_plugs):
                # making a copy of the plug object prevents a maya crash
                plugs[i] = OpenMaya.MPlug(plug_array[i])

            self._plugs.update(
                {plug.partialName(useLongNames=True): plug for plug in plugs}
            )

            return plugs

    def list_attributes(self, extend_to_shape: bool = False) -> List["MirageAttr"]:
        """Gets a list of all attributes on this node as MirageAttrs.


        Args:
            extend_to_shape (bool): If True, then listed attributes will
                include those on the transform's Shape node.
        """
        plugs = self.all_plugs
        if extend_to_shape:
            try:
                shape = self.shape
            except TypeError:
                pass
            else:
                if shape and shape != self:
                    plugs.extend(shape.all_plugs)
        return [MirageAttr(plug) for plug in plugs]

    def list_connections(
        self, mode: ConnectionTypes = ConnectionTypes.DESTINATION, extend_to_shape=True
    ) -> List["MirageAttr"]:
        """Gets a list of all connections on this node as MirageAttrs.

        Args:
            mode (enum): One of the enumerated types in ConnectionType, either
                source, destination, or all. If no mode is provided, defaults
                to destination.
            extend_to_shape (bool): If True, then listed attributes will
                include those on the transform's Shape node.
        """

        plugs = self.all_connected_plugs

        if extend_to_shape:
            try:
                shape = self.shape
            except TypeError:
                pass
            else:
                if shape and shape != self:
                    plugs.extend(shape.all_connected_plugs)
        if mode == ConnectionTypes.DESTINATION:
            return [MirageAttr(plug) for plug in plugs if plug.isDestination]
        elif mode == ConnectionTypes.SOURCE:
            return [MirageAttr(plug) for plug in plugs if plug.isSource]
        elif mode == ConnectionTypes.ALL:
            return [MirageAttr(plug) for plug in plugs]
        return []

    @property
    def inputs(self) -> List["MirageAttr"]:
        """List of the attributes which are connected from another node."""
        return self.list_connections(mode=ConnectionTypes.DESTINATION)

    @property
    def input_connections(self) -> List[Tuple["MirageAttr", "MirageAttr"]]:
        """List of attribute connections and the nodes they connect from.

        Returns:
            A list of tuples, where the first item is the connected attribute
            on the other node, and the second item is the attribute on this
            node.
        """
        return [(inp.value, inp) for inp in self.inputs]  # type: ignore

    @property
    def outputs(self) -> List["MirageAttr"]:
        """List of attributes which are connected to another node."""
        return self.list_connections(mode=ConnectionTypes.SOURCE)

    @property
    def output_connections(self) -> List[Tuple["MirageAttr", "MirageAttr"]]:
        """List of attribute connections and the nodes they connect to."""
        return [(outp.value, outp) for outp in self.outputs]  # type: ignore

    @property
    def attributes(self) -> List["MirageAttr"]:
        """List of attributes on this node."""
        return self.list_attributes(extend_to_shape=False)

    @property
    def attribute_names(self) -> List[str]:
        """List of maya-friendly attribute names for this node."""
        return [attr.name for attr in self.attributes]

    @property
    def dynamic_attributes(self) -> List["MirageAttr"]:
        """List of only the dynamic attributes on this node."""
        return [attr for attr in self.attributes if attr.is_dynamic]

    def has_attribute(self, attribute_as_string: str) -> bool:
        """True if the given attribute exists on this node.
        Args:
            attribute_as_string (str): The name of an attribute to query.
        """
        # The cmds version of this query is faster and easier than the
        # equivalent in OpenMaya.
        return attribute_as_string in cmds.listAttr(self.long_name)

    # ATTRIBUTE CREATION

    def _attr_creator(
        self,
        long_name: str,
        short_name: Optional[str] = None,
        initial_value: Optional[Any] = None,
    ) -> "AttrCreationContext":
        """Returns an attribute creation context manager for this node.

        These context managers do the following:

        1. translate long names into short names if needed
        2. check to see if the long or short names are already in use
        3. add the dependency graph modifier and execute it

        See the AttrCreationContext class for more information.
        """
        return AttrCreationContext(self, long_name, short_name, initial_value)

    def add_vector_attr(
        self,
        long_name: str,
        short_name: Optional[str] = None,
        default_value: Optional[Iterable[float]] = None,
        initial_value: Optional[Iterable[float]] = None,
    ) -> "MirageAttr":
        """Creates a compound attribute of three Doubles.

        Child attributes will have X, Y and Z appended to both the long and
        short names. This method mirrors the behavior of the Add Attribute
        menu option in the attribute editor in Maya.

        Args:
            long_name (str): The long name of the attribute. By convention,
                this should be in camelCase.

            short_name (str): The short name of the attribute, typically no
                more than three characters.  If no short name is provided, the
                initials of the long name will be used.

            default_value: The value that will be set as the default on the
                attribute.

            initial_value: The value that will be set right after creating the
                new attribute.  This is distinct from the default value.

        Returns: MirageAttr
        """
        attr_ctx = self._attr_creator(long_name, short_name, initial_value)
        with attr_ctx as attr_creator:
            m_attr = _make_vector_attr(
                long_name, attr_creator.short_name, default_value
            )
            attr_creator.m_attr = m_attr
        return attr_creator.mirage_attr

    def add_string_attr(
        self,
        long_name: str,
        short_name: Optional[str] = None,
        default_value: Optional[str] = None,
        initial_value: Optional[str] = None,
    ):
        """Creates a string attribute on this node.

        Args:
            long_name (str): The long name of the attribute. By convention,
                this should be in camelCase.

            short_name (str): The short name of the attribute, typically no
                more than three characters.  If no short name is provided, the
                initials of the long name will be used.

            default_value: The value that will be set as the default on the
                attribute.

            initial_value: The value that will be set right after creating the
                new attribute.  This is distinct from the default value.
        """
        attr_ctx = self._attr_creator(long_name, short_name, initial_value)
        with attr_ctx as attr_creator:
            m_attr = _make_string_attr(
                long_name, attr_creator.short_name, default_value
            )
            attr_creator.m_attr = m_attr
        return attr_creator.mirage_attr

    def _add_simple_numeric_attr(
        self,
        data_type: OpenMaya.MFnNumericData,
        long_name: str,
        short_name: Optional[str] = None,
        default_value: Optional[Union[float, int]] = None,
        initial_value: Optional[Union[float, int]] = None,
    ) -> "MirageAttr":
        """Creates a numeric attribute of the given data type"""
        attr_ctx = self._attr_creator(long_name, short_name, initial_value)
        with attr_ctx as attr_creator:
            m_attr = make_numeric_attr(
                long_name,
                attr_creator.short_name,
                data_type,
                default_value=default_value,
            )
            attr_creator.m_attr = m_attr
        return attr_creator.mirage_attr

    def add_int_attr(
        self,
        long_name: str,
        short_name: Optional[str] = None,
        default_value: Optional[int] = None,
        initial_value: Optional[int] = None,
    ) -> "MirageAttr":
        """Creates an integer attribute on this node.

        Args:
            long_name (str): The long name of the attribute. By convention,
                this should be in camelCase.

            short_name (str): The short name of the attribute, typically no
                more than three characters.  If no short name is provided, the
                initials of the long name will be used.

            default_value: The value that will be set as the default on the
                attribute.

            initial_value: The value that will be set right after creating the
                new attribute.  This is distinct from the default value.
        """
        data_type = OpenMaya.MFnNumericData.kInt
        return self._add_simple_numeric_attr(
            data_type, long_name, short_name, default_value, initial_value
        )

    def add_float_attr(
        self,
        long_name: str,
        short_name: Optional[str] = None,
        default_value: Optional[float] = None,
        initial_value: Optional[float] = None,
    ) -> "MirageAttr":
        """Creates a floating point attribute on this node.

        Args:
            long_name (str): The long name of the attribute. By convention,
                this should be in camelCase.

            short_name (str): The short name of the attribute, typically no
                more than three characters.  If no short name is provided, the
                initials of the long name will be used.

            default_value: The value that will be set as the default on the
                attribute.

            initial_value: The value that will be set right after creating the
                new attribute.  This is distinct from the default value.
        """
        data_type = OpenMaya.MFnNumericData.kFloat
        return self._add_simple_numeric_attr(
            data_type, long_name, short_name, default_value, initial_value
        )

    def add_bool_attr(
        self,
        long_name: str,
        short_name: Optional[str] = None,
        default_value: Optional[bool] = None,
        initial_value: Optional[bool] = None,
    ) -> "MirageAttr":
        """Creates a boolean attribute on this node.

        Args:
            long_name (str): The long name of the attribute. By convention,
                this should be in camelCase.

            short_name (str): The short name of the attribute, typically no
                more than three characters.  If no short name is provided, the
                initials of the long name will be used.

            default_value: The value that will be set as the default on the
                attribute.

            initial_value: The value that will be set right after creating the
                new attribute.  This is distinct from the default value.
        """
        data_type = OpenMaya.MFnNumericData.kBoolean
        return self._add_simple_numeric_attr(
            data_type, long_name, short_name, default_value, initial_value
        )

    def add_color_attr(
        self,
        long_name: str,
        short_name: Optional[str] = None,
        initial_value: Optional[Iterable[float]] = None,
    ) -> "MirageAttr":
        """Creates a color compound attribute on this node.

        Child attributes will be Floats with "r", "g", and "b" appended to both
        their long and short names.

        NOTE: Color attributes will always default to r, g, and b values of 0,
            and other defaults cannot be provided here.  This and the naming
            mirrors the behavior of the maya UI

        Args:
            long_name (str): The long name of the attribute. By convention,
                this should be in camelCase.

            short_name (str): The short name of the attribute, typically no
                more than three characters.  If no short name is provided, the
                initials of the long name will be used.

            initial_value: The value that will be set right after creating the
                new attribute.  This is distinct from the default value.
        """
        attr_ctx = self._attr_creator(long_name, short_name, initial_value)
        with attr_ctx as attr_creator:
            attr_creator.m_attr = _fn_numeric_attr().createColor(
                long_name, attr_creator.short_name
            )
        return attr_creator.mirage_attr

    def add_enum_attr(
        self,
        long_name: str,
        short_name: Optional[str] = None,
        default_value: Optional[int] = None,
        initial_value: Optional[int] = None,
        fields: Optional[Iterable[str]] = None,
    ) -> "MirageAttr":
        """Create an enumeration attribute on this node.

        NOTE: all "values" on enumerated attributes are integers representing
            indexes of the enumeration.  So, if you had fields=["a", "b", "c"]
            and you want "b" to be the default, you'd use default_value=1

        Args:
            long_name (str): The long name of the attribute. By convention,
                this should be in camelCase.

            short_name (str): The short name of the attribute, typically no
                more than three characters.  If no short name is provided, the
                initials of the long name will be used.

            default_value: The value that will be set as the default on the
                attribute.

            initial_value: The value that will be set right after creating the
                new attribute.  This is distinct from the default value.

            fields (iterable): an iterable of strings that will be the fields
            available to the attribute by default.
        """
        attr_ctx = self._attr_creator(long_name, short_name, initial_value)
        with attr_ctx as attr_creator:
            fn_set = _fn_enum_attr()
            attr_creator.m_attr = _make_attr(
                long_name,
                attr_creator.short_name,
                fn_set.create,
                default_value=default_value,
            )
            if fields:
                for i, field in enumerate(fields):
                    fn_set.addField(field, i)
        return attr_creator.mirage_attr

    # SHADING

    @property
    def shading_group(self) -> "MirageNode":
        """Attempts to get the shading group of this node.

        If this property is accessed from a non-shade-able node, an error will
        be raised.
        """
        try:
            dests = [a.mirage_node for a in self["instObjGroups"][0].destinations]
        except Exception:
            raise TypeError(
                "Error raised while attempting to query a shading group on "
                "{name}.  This node might not be shaded!"
                "".format(name=self.name)
            )
        else:
            for dest in dests:
                if dest.api_type == "kShadingEngine":
                    return dest
        raise TypeError(f"Given node {self.name} is not shaded!")

    @shading_group.setter
    def shading_group(self, shading_group):
        """Attempts to set the shading group of this node.

        If this property is set from a non-shade-able node, an error will be
        raised.
        """
        self.shading_group.fn_set.removeMember(self._m_obj)
        shading_group.fn_set.addMember(self._m_obj)

    # SET-LIKE OBJECT INTERFACE

    def add_set_member(self, mirage_node: "MirageNode"):
        """Adds a member MirageNode to this set-like node."""
        self.fn_set.addMember(mirage_node._m_obj)

    def add_set_members(self, mirage_nodes: Iterable["MirageNode"]):
        """Adds an iterable of MirageNode members to this set-like node."""

        sel_list = OpenMaya.MSelectionList()
        for mirage_node in mirage_nodes:
            # The MDagPath is required here rather than the MObject
            # representation. The MObject breaks some internal connections
            # for unknown reasons. Functionally, they should be identical.
            sel_list.add(mirage_node._m_dag_path)

        self.fn_set.addMembers(sel_list)

    def is_set_member(self, mirage_node: "MirageNode"):
        """Returns True if a given node is a member of this set-like node."""
        return self.fn_set.isMember(mirage_node._m_obj)

    def remove_set_member(self, mirage_node: "MirageNode"):
        """Removes a member MirageNode from this set-like node."""
        self.fn_set.removeMember(mirage_node._m_obj)

    def remove_set_members(self, mirage_nodes: Iterable["MirageNode"]):
        """Removes an iterable of MirageNodes from this set-like node."""
        for mirage_node in mirage_nodes:
            self.remove_set_member(mirage_node)

    def clear_set(self):
        """Empties out the membership of this set-like node."""
        self.fn_set.clear()

    @property
    def set_membership(self) -> List["MirageNode"]:
        """A list of members of this set-like node."""
        sel_list = self.fn_set.getMembers(False)
        return [
            MirageNode.from_m_obj(sel_list.getDependNode(i))
            for i in range(sel_list.length())
        ]

    @set_membership.setter
    def set_membership(self, mirage_nodes: Iterable["MirageNode"]):
        """Assigns an iterable of MirageNodes to this node's membership.

        This method clears out the previous membership.
        """
        self.clear_set()
        self.add_set_members(mirage_nodes)

    @property
    def set_membership_flattened(self) -> List["MirageNode"]:
        """A list of this node's members, flattening out any nested sets."""
        sel_list = self.fn_set.getMembers(True)
        return [
            MirageNode.from_m_obj(sel_list.getDependNode(i))
            for i in range(sel_list.length)
        ]

    # QUATERNION ROTATIONS

    def set_quaternion(self, quat_len_4_iterable: Iterable[float]):
        """Sets the rotation of this node, using a quaternion.

        Args:
            quat_len_4_iterable (iterable): A 4-length collection of floats
                representing a quaternion rotation.
        """
        self.fn_transform.setRotationComponents(
            quat_len_4_iterable, OpenMaya.MSpace.kObject, asQuaternion=True
        )

    # MAGIC

    def __repr__(self):
        """Returns the code representation of this node:
        <MirageNode myNode at 0x0000000>
        """
        return "<MirageNode {name} at {hex}>".format(name=self.name, hex=hex(id(self)))

    def __getitem__(self, attr_name: str) -> "MirageAttr":
        """Queries the attribute using dictionary-style lookup.

        Example::
            mirage_attr_obj = my_node["someAttribute"]
        """
        return self.attr(attr_name)

    def __setitem__(self, attr_name: str, value: Any):
        """Set the value of the attribute directly dictionary-style.

        Example:
            my_node["someAttribute"] = 5.0
        """
        attr = self.attr(attr_name)
        if isinstance(value, MirageAttr):
            attr.connect_from(value)
            return
        attr.value = value

    def __hash__(self):
        return self.hash_code

    def __eq__(self, other: "MirageNode"):
        try:
            return self.hash_code == other.hash_code
        except Exception:
            return False


class MirageAttr:
    """MirageAttr is an interface to Maya attribute objects and function sets.

    Creating MirageAttrs:
    ---------------------

    By default, MirageAttrs are instantiated using a maya MPlug object.

    Also, alternate constructors are provided to instantiate MirageAttrs:

    Using the full name of an attribute::

        attr = MirageAttr.from_full_name("my_node.tx")

    Using an existing MirageNode and an attribute name::

        attr = MirageAttr.from_mirage_node_and_name(my_mirage_node, "tx")

    Given a MirageNode, MirageAttrs can be accessed through their names, using
    either the "attr" method or dictionary-style lookup (__getitem__)::

        attr = my_mirage_node.attr("someAttribute")
        attr = my_mirage_node["someAttribute"]

    """

    def __init__(self, plug: Optional[OpenMaya.MPlug] = None):
        """
        Args:
            plug (MPlug): A maya MPlug object representing the attribute.

        Attributes:
            getter_setter (AttrGetterSetterMap): An object used to obtain the
                appropriate getter and setter functions to for this attribute.
        """
        if plug is None:
            self._m_plug: OpenMaya.MPlug
        else:
            self._m_plug = plug
        self._mirage_node = None
        self.getter_setter = AttrGetterSetterMap(self)

    # ALTERNATE CONSTRUCTORS

    @classmethod
    def from_full_name(cls, full_attribute_name: str) -> "MirageAttr":
        """Uses a dot-separated string to create a new MirageAttr."""
        node_name, attr_name = full_attribute_name.split(".")

        m_obj = m_obj_from_name(node_name)
        mirage_node = MirageNode.from_m_obj_cached(m_obj)

        return cls.from_mirage_node_and_name(mirage_node, attr_name)

    @classmethod
    def from_mirage_node_and_name(
        cls, mirage_node: MirageNode, attr_name: str
    ) -> "MirageAttr":
        """Uses a MirageNode and an attribute name to make a new MirageAttr."""
        try:
            return mirage_node._attrs[attr_name]
        except Exception:
            pass
        instance = cls(mirage_node.plug(attr_name))
        instance._mirage_node = mirage_node
        mirage_node._attrs[attr_name] = instance
        return instance

    # FUNCTION SETS

    @property
    def fn_attr(self) -> OpenMaya.MFnAttribute:
        """The MFnAttribute object for this attribute"""
        return _fn_attr_for_attribute(self.attribute)

    # LAZY PROPERTIES

    @property
    def mirage_node(self) -> MirageNode:
        """The MirageNode for this attribute"""
        if self._mirage_node is not None:
            return self._mirage_node
        mirage_node = MirageNode.from_m_obj_cached(self._m_plug.node())
        self._mirage_node = mirage_node
        return mirage_node

    @property
    def attribute(self) -> OpenMaya.MObject:
        """The MObject object for this attribute."""
        return self._m_plug.attribute()

    @property
    def name(self) -> str:
        """The name of this attribute, not including the name of the node."""
        return self._m_plug.partialName(useLongNames=True)

    @property
    def full_name(self) -> str:
        """The name of this attribute, including the name of the node.
        The node name and the attribute name will be separated by a dot.

        Example::
            my_attr.full_name
            'some_node.myAttr'
        """
        return str(self._m_plug.name())

    @property
    def attr_type(self) -> str:
        """The attribute type.

        Attribute type is a category that defines the types of data that are
        valid for this attribute.
        """
        return get_attr_type(self.attribute)

    @property
    def is_array(self) -> bool:
        """True if the attribute's plug is an array"""
        return self._m_plug.isArray

    @property
    def is_compound(self) -> bool:
        """True if the attribute's plug is compound"""
        return self._m_plug.isCompound

    @property
    def is_destination(self) -> bool:
        """True if the attribute's plug is connected from another plug."""
        return self._m_plug.isDestination

    @property
    def data_type(self) -> str:
        """The data type of the attribute."""
        return get_data_type(self.attribute, self.attr_type)

    @property
    def is_readable(self) -> bool:
        """True if the attribute is readable."""
        return self.fn_attr.readable

    @property
    def is_writable(self) -> bool:
        """True if the attribute is not read-only."""
        return self.fn_attr.writable

    @property
    def is_connectable(self) -> bool:
        """True if the attribute can be connected to other attributes"""
        return self.fn_attr.connectable

    @property
    def is_hidden(self) -> bool:
        """True if the attribute is not shown in the Attribute Editor"""
        return self.fn_attr.hidden

    @property
    def is_dynamic(self) -> bool:
        """True if the attribute is dynamic.

        Dynamic is a term of art in Maya that means the attribute was created
        outside of the node definition, i.e. it was added by a user or process
        during a Maya session.
        """
        return self.fn_attr.dynamic

    @property
    def is_keyable(self) -> bool:
        """True if the attribute accepts animation curves"""
        return self.fn_attr.keyable

    @property
    def is_storable(self) -> bool:
        """True if the attribute is storable via MEL.

        Some attributes only exist during a Maya session, or are calculated
        from other attributes.
        """
        return self.fn_attr.storable

    # ATTRIBUTE VALUES AND EVALUATION

    @property
    def is_default_value(self) -> bool:
        """True if the attribute's current value is the same as its default."""
        return self._m_plug.isDefaultValue()

    @property
    def value(self) -> Any:
        """The current value of this MirageAttr's associated MPlug object.

        If the MPlug is a network plug and is connected to another attribute,
        the MirageAttr for the source connection is returned. Otherwise, the
        static value of the attribute is returned.

        NOTE: currently, the MirageAttr system does not have an interface for
            working with keyframed attribute values.  This functionality will
            be added at a later date.
        """

        _m_plug = self._m_plug

        if self._m_plug.isDestination:
            return get_connection(_m_plug)

        data_type = self.data_type

        is_array = self.is_array

        if is_array and data_type == "kInvalid":
            return get_array(_m_plug)

        if is_array and data_type not in self.getter_setter:
            return get_array(_m_plug)

        if self.is_compound:
            return get_compound(_m_plug)

        try:
            return self.getter_setter.getter(_m_plug)
        except RuntimeError as e:
            try:
                return cmds.getAttr(self.full_name)
            except Exception:
                raise e

    @value.setter
    def value(self, value: Any):
        """Sets the value of the attribute.

        For compound attributes, iterables with the correct length and
        nesting characteristics must be passed in as the value.  For example::

            translate_attr.value = [1, 2, 3]
            uv_attr.value = [0, 1]

        For array attributes, iterables of any length can be provided::

            # make a rainbow ramp!
            ramp["colorEntryList"].value = [0.0, [1.0, 0.0, 0.0],
                                            0.5, [0.0, 1.0, 0.0],
                                            1.0, [0.0, 0.0, 1.0]]

        """
        is_array = self.is_array

        if is_array and self.data_type == "kMatrix":
            return _set_matrix(self._m_plug, value)
        elif is_array:
            return set_array(self._m_plug, value)
        elif self.is_compound:
            return set_compound(self._m_plug, value)
        else:
            return self.getter_setter.setter(self._m_plug, value)

    # CONNECTIONS

    def connect_to(self, mirage_attr: "MirageAttr", force: bool = True):
        """Connects the plug from this MirageAttr (the source) to another
        MirageAttr (the destination).

        Args:
            mirage_attr (MirageAttr): The destination attribute.

            force (bool): When True, any existing connections on the
                destination attribute will be removed first.
        """
        make_connection(self._m_plug, mirage_attr._m_plug, force=force)

    def connect_from(self, mirage_attr: "MirageAttr", force: bool = True):
        """Connects from another mirage_attr (the source) to this MirageAttr
        (the destination).

        Args:
            mirage_attr (MirageAttr): The destination attribute.

            force (bool): When True, any existing connections on the
                destination attribute will be removed first.

        """
        make_connection(mirage_attr._m_plug, self._m_plug, force=force)

    def disconnect(self):
        """Breaks any source connections on this attribute."""
        break_source_connection(self._m_plug)

    @property
    def destinations(self) -> List["MirageAttr"]:
        """A list of MirageAttrs that receive this attribute as an input."""
        dests = self._m_plug.destinations()
        return [MirageAttr(dest) for dest in dests]

    # MAGIC

    def __getitem__(self, item: Union[str, int]) -> "MirageAttr":
        """Gets the named or indexed child attribute.

        This is valid for both compound and array attributes::

            translate_x = mirage_node["translate"]["X"]
            translate_y = mirage_node["translate"][1]

        This method also works for weird logical indices that maya uses for
        certain attributes, such as "worldInverseMatrix[-1]"
        """

        m_plug = self._m_plug

        # item is an index
        if self.is_array:
            return MirageAttr(m_plug.elementByLogicalIndex(item))

        # item is either an index or a string representing a child attribute
        elif self.is_compound:
            try:
                return MirageAttr(OpenMaya.MPlug(m_plug.child(item)))

            except NotImplementedError:
                # assume the "item" is a string representing a name
                for i in range(m_plug.numChildren()):
                    child_plug = m_plug.child(i)
                    if child_plug.name().endswith(item):
                        return MirageAttr(child_plug)
                else:
                    raise ValueError(f"Attribute has no child {item}!")
        else:
            raise ValueError("Attribute has no children!")

    def __setitem__(self, attr: Union[str, int], value: Any):
        """Sets the value of the attribute."""
        child_attr = self[attr]
        if isinstance(value, MirageAttr):
            child_attr.connect_from(value)
            return
        child_attr.value = value

    def __repr__(self):
        """Returns the code representation of the attribute.

        <MirageAttr my_node.someAttribute (kDataType) at 0x0000000>
        """

        # The try-except blocks below prevent logging and diagnostic
        # tools from failing in weird edge cases.
        try:
            name = self.full_name
        except Exception:
            name = "UnknownAttr"
        try:
            attr_type = self.attr_type
        except Exception:
            attr_type = "UnknownAttrType"
        try:
            data_type = self.data_type
        except Exception:
            data_type = "UnknownDataType"

        return "<MirageAttr {name}, {attr_type} ({data_type}) at {hex_id}>" "".format(
            name=name, attr_type=attr_type, data_type=data_type, hex_id=hex(id(self))
        )

    def __eq__(self, other: "MirageAttr") -> bool:
        """Returns True if the attribute name and target node are the same."""
        return isinstance(other, self.__class__) and self.full_name == other.full_name


# ATTRIBUTE CREATION


def _make_attr(
    long_name: str,
    short_name: Optional[str],
    attr_constructor: Callable,
    data_type: Optional[str] = None,
    default_value: Optional[Any] = None,
) -> "AttrCreationContext":
    """Makes an OpenMaya Attribute object to assign to a node.

    Args:
        long_name (str): The long name of the attribute, typically camelCase.

        short_name (str): the short name of the attribute, typically just a
            couple of initials.

        attr_constructor (callable): A function that will actually build the
            attribute object.

        data_type (str): A maya-friendly data type.

        default_value: A value that will be set as the attributes default.
    """
    # to account for the various function signatures available for different
    # attribute and data types, we build an additional argument list here
    args = list()

    if data_type is not None:
        args.append(data_type)

    if default_value is not None:
        args.append(default_value)

    return attr_constructor(long_name, short_name, *args)


def make_numeric_attr(
    long_name: str,
    short_name: str,
    data_type: Optional[str],
    default_value: Optional[Union[float, int]] = None,
) -> "AttrCreationContext":
    """Makes a numeric attribute with a data type from MFnNumericAttribute.

    Args:
        long_name (str): The long name of the attribute, typically camelCase.

        short_name (str): the short name of the attribute, typically just a
            couple of initials.

        data_type (str): A maya-friendly data type.

        default_value: A value that will be set as the attribute's default.
    """

    constructor = _fn_numeric_attr().create
    return _make_attr(long_name, short_name, constructor, data_type, default_value)


def _make_numeric_triple_attr(
    long_name: str,
    short_name: str,
    axes: Iterable[str],
    data_type: str,
    default_value: Optional[Iterable[Union[float, int]]] = None,
) -> OpenMaya.MObject:
    """Makes a three-child compound numeric attribute.

    This is useful for colors (R, G, B) or transforms (X, Y, Z).

    Args:
        long_name (str): The long name of the attribute, typically camelCase.

        short_name (str): the short name of the attribute, typically just a
            couple of initials.

        axes (iterable): A three-length collection of strings representing the
            labels of each child attribute.

        data_type (str): A maya-friendly data type.

        default_value (iterable): A value that will be set as the attribute's
            default.

    Example::

            attr = _make_numeric_triple_attr(
                "longName", "ln", ["X", "Y", "Z"], "kDouble"
            )
    """
    default_value = default_value or [0.0, 0.0, 0.0]
    children = list()
    for value, axis in zip(default_value, axes):
        child_long_name = "{long_name}{axis}".format(long_name=long_name, axis=axis)
        child_short_name = "{short_name}{axis}".format(short_name=short_name, axis=axis)
        children.append(
            make_numeric_attr(child_long_name, child_short_name, data_type, value)
        )
    return _fn_numeric_attr().create(long_name, short_name, *children)


def _make_vector_attr(
    long_name: str, short_name: str, default_value: Optional[Iterable[float]] = None
) -> OpenMaya.MObject:
    """Makes a three-child compound numeric attribute of kDoubles.

    "X", "Y", and "Z" will be appended to both the long name and the
    short name for each child.

    Args:
        long_name (str): The long name of the attribute, typically camelCase.

        short_name (str): the short name of the attribute, typically just a
            couple of initials.

        default_value (iterable): A value that will be set as the attribute's
            default. This must be a three-length iterable.
    """
    axes = ["X", "Y", "Z"]
    data_type = OpenMaya.MFnNumericData.kDouble
    return _make_numeric_triple_attr(
        long_name, short_name, axes, data_type, default_value=default_value
    )


def _make_string_attr(
    long_name: str, short_name: str, default_value: Optional[str] = None
) -> OpenMaya.MObject:
    """Makes a string attribute with its corresponding required data block.

    Args:
        long_name (str): The long name of the attribute, typically camelCase.

        short_name (str): the short name of the attribute, typically just a
            couple of initials.

        default_value (str): A value that will be set as the attribute's
            default.
    """
    default_value = default_value or ""
    fn_set = _fn_typed_attr()
    data_block = OpenMaya.MFnStringData()
    data_obj = data_block.create()
    data_block.set(default_value)
    return fn_set.create(long_name, short_name, OpenMaya.MFnData.kString, data_obj)


# ATTRIBUTE GETTER AND SETTERS


def get_connection(m_plug: OpenMaya.MPlug) -> MirageAttr:
    """Returns a MirageAttr that is connected to the given plug."""
    m_plug_array = m_plug.connectedTo(True, False)

    # copying the MPlug prevents a maya crash
    return MirageAttr(OpenMaya.MPlug(m_plug_array[0]))


def get_array(m_plug: OpenMaya.MPlug) -> list:
    """Gets the values stored in an array attribute.

    Array Attributes are collections of attributes sharing the same data
    type.  Child attributes can be either connected or statically-driven.
    """
    elem_idxs = m_plug.getExistingArrayAttributeIndices()

    idx_len = len(elem_idxs)
    all_element_values = [Any] * idx_len

    for i in range(idx_len):
        try:
            element_plug = m_plug.elementByLogicalIndex(i)
        except RuntimeError:
            # Translate Maya's runtime error into an AttributeError
            msg = "Element {number} on {plug_name} is not available" "".format(
                number=i, plug_name=m_plug.name()
            )
            raise AttributeError(msg)

        # copying the MPlug prevents a maya crash
        new_att = MirageAttr(element_plug)
        all_element_values[i] = new_att.value

    return all_element_values


def set_array(m_plug: OpenMaya.MPlug, value: Iterable):
    """Sets the value of an array attribute.
    Args:
        value (iterable): The value that will be set on the array attribute.
            This value is an iterable of any length, and will always set the
            array's child values in logical order.
    """
    for i, ith_value in enumerate(value):
        attr = MirageAttr(m_plug.elementByLogicalIndex(i))
        attr.value = ith_value


def get_compound(m_plug: OpenMaya.MPlug) -> List:
    """Gets the value of a compound attribute.

    Compound Attributes can have multiple data types and can also have a mix
    of connected and non-connected plugs.
    """
    compound_len = m_plug.numChildren()
    all_element_values = [Any] * compound_len
    for i in range(compound_len):
        child_plug = m_plug.child(i)
        new_att = MirageAttr(child_plug)
        all_element_values[i] = new_att.value
    return all_element_values


def set_compound(m_plug: OpenMaya.MPlug, value: Iterable):
    """Sets the values of all the children of the compound MPlug.

    If any of the children are also compound attributes, this function will
    recurse

        Args:
        value (iterable): The value that will be set on the array attribute.
            This value is an iterable whose length matches the attribute's
            child count.

    """
    enumerated_value = enumerate(value)
    for i, ithvalue in enumerated_value:
        attr = MirageAttr(m_plug.child(i))
        attr.value = ithvalue
    return


def _get_matrix(plug: OpenMaya.MPlug) -> List[List[float]]:
    """Getter for Matrix data types.

    Returns:
        A nested set of four lists with four elements each, representing the
        4x4 matrix (Maya only supports 4x4 matrices).

    Example::
        _get_matrix(plug)
        result: [[1, 0, 0, 0]
                 [0, 1, 0, 0]
                 [0, 0, 1, 0]
                 [0, 0, 0, 0]]
    """

    # some matrix attributes are nested in a single-element array.  I assume
    # this is meant to support some kind of multi-matrix attribute type, but
    # have not seen it used this way in practice.  We always only parse the
    # first element of the array.
    if plug.isArray:
        plug = plug.elementByLogicalIndex(0)

    matrix = OpenMaya.MFnMatrixData(plug.asMObject()).matrix()

    # NOTE: the getElement method of MMatrix objects is only in Python, the
    #       C++ equivalent uses a different approach.
    return [[matrix.getElement(i, j) for i in range(4)] for j in range(4)]


def _set_matrix(m_plug: OpenMaya.MPlug, value: Sequence[Sequence[float]]):
    """Setter for Matrix data types.

    Args:
        value (sequence): A nested set of four lists with four elements each,
            representing the 4x4 matrix (Maya only supports 4x4 matrices).
    """
    # some matrix attributes are nested in a single-element array.
    if m_plug.isArray:
        m_plug = m_plug.elementByLogicalIndex(0)

    matrix = OpenMaya.MMatrix()
    for i in range(4):
        for j in range(4):
            # NOTE: the setElement method of MMatrix objects is only in Python,
            #       the C++ equivalent uses a different approach.
            matrix.setElement(i, j, value[i][j])
    new_mobject = OpenMaya.MFnMatrixData().create(matrix)
    m_plug.setMObject(new_mobject)


def _get_angle(m_plug: OpenMaya.MPlug) -> float:
    """Getter for Angle data types.

    We always work in degrees when handling angular values for convenience and
    consistency. If you need to work in Radians for any reason, convert to
    degrees before using the functions in mirage.
    """
    m_angle_obj = m_plug.asMAngle()
    return float(m_angle_obj.asDegrees())


def _get_message(m_plug: OpenMaya.MPlug) -> Union[MirageAttr, List[MirageAttr]]:
    """Getter for message attributes.

    Message attributes are used in Maya to logically link entire nodes together
    in various ways. These attributes are complicated because they don't
    contain data, and therefore have no associated data type. A message-typed
    attribute can either be a source or a destination (this node.message is a
    member of that node's membership.messages) Destination message attributes
    can also be arrays or compounds (or a mix of both), so we have to check
    against both configurations.

    When a source-type message attribute's value is queried, the attribute
    itself is returned.

    When a destination-type message attribute's value is queried, a list of
    source-type MirageAttrs is returned.

    Example::

        print(my_node["message"].value)
        result:
            <MirageAttr my_node.message>

        print(another_node["messages"].value)
        result:
            [<MirageAttr my_node.message>, <MirageAttr a_third_node.message>]
    """

    if m_plug.isArray:
        return get_array(m_plug)

    elif m_plug.isCompound:
        return get_compound(m_plug)

    elif m_plug.isDestination:
        return get_connection(m_plug)

    return MirageAttr(m_plug)


def _get_time(m_plug: OpenMaya.MPlug) -> float:
    """Getter for time attributes.

    Returns:
        A time value in whatever Maya's current time-units are.
    """
    return m_plug.asMTime().value


def _get_invalid(_):
    """Getter for internal or hidden attributes in Maya.

    Access to these attributes is unscriptable and therefore we just obtain
    None whenever one of these attributes is accessed.
    """
    return None


def _set_invalid(m_plug: OpenMaya.MPlug, *_):
    """Setter for internal or hidden attributes in Maya.

    We just raise an error when an invalid attribute type is assigned to
    """
    raise AttributeError(
        "{name} is an invalid attribute to assign values to"
        "".format(name=m_plug.name())
    )


class AttrGetterSetterMap:
    """A convenient interface to the Maya Api's getters and setters.

    When instantiated with an MirageAttr, this object provides the best-match
    getter and setter for the MirageAttr to use when values are queried or
    assigned to
    """

    _mplug = OpenMaya.MPlug
    get_bool, set_bool = _mplug.asBool, _mplug.setBool
    get_int, set_int = _mplug.asInt, _mplug.setInt
    get_float, set_float = _mplug.asFloat, _mplug.setFloat
    get_double, set_double = _mplug.asDouble, _mplug.setDouble
    get_string, set_string = _mplug.asString, _mplug.setString
    get_char, set_char = _mplug.asChar, _mplug.setChar
    get_time, set_time = _get_time, None
    get_message, set_message = _get_message, None
    get_matrix, set_matrix = _get_matrix, _set_matrix
    get_angle, set_angle = _get_angle, None

    # invalid data types are used as placeholders for non-data attributes.
    # We just return None when querying the value of invalid data types.
    # We also use these functions for the kAny and kGenericAttribute data
    # types which are also only used for placeholders
    get_none, set_none = _get_invalid, _set_invalid

    # convenient lookup table for getters and setters for MPlugs of various
    # numeric and other types
    typed_getter_setters = {
        "kBoolean": (get_bool, set_bool),
        "kByte": (get_int, set_int),
        "kLong": (get_int, set_int),
        "kInt": (get_int, set_int),
        "k2Int": (get_int, set_int),
        "k3Int": (get_int, set_int),
        "kInt64": (get_int, set_int),
        "kIntArray": (get_int, set_int),
        "kShort": (get_int, set_int),
        "kEnumAttribute": (get_int, set_int),
        "kFloat": (get_float, set_float),
        "k2Float": (get_float, set_float),
        "k3Float": (get_float, set_float),
        "kUnitAttribute": (get_float, set_float),
        "kDistance": (get_double, set_double),
        "kDouble": (get_double, set_double),
        "k2Double": (get_double, set_double),
        "k3Double": (get_double, set_double),
        "k4Double": (get_double, set_double),
        "kAddr": (get_double, set_double),
        "kString": (get_string, set_string),
        "kStringArray": (get_string, set_string),
        "kChar": (get_char, set_char),
        "kInvalid": (get_none, set_none),
        "kAny": (get_none, set_none),
        "kGenericAttribute": (get_none, set_none),
        "kMatrix": (get_matrix, set_matrix),
        "kMatrixAttribute": (get_matrix, set_matrix),
        "kTime": (get_time, set_time),
        "kAngle": (get_angle, set_angle),
        "kMessageAttribute": (get_message, set_message),
    }

    def __init__(self, mirage_attribute_instance: MirageAttr):
        """
        Attributes:
            mirage_attr (MirageAttr): The underlying attribute that we want to
                obtain getters and setters for.
        """
        self.mirage_attr = mirage_attribute_instance

    @property
    def getter(self) -> Callable:
        """Get the appropriate "getter" for the MirageAttr's data type.

        The getter is always a callable that takes a maya MPlug object as its
        only argument.
        """
        return self[self.mirage_attr.data_type][0]

    @property
    def setter(self) -> Callable:
        """Get the appropriate "setter" for the MirageAttr's data type.

        The setter is always a callable that takes a maya MPlug object and a
        value as its arguments.
        """
        return self[self.mirage_attr.data_type][1]

    def __contains__(self, data_type: str) -> bool:
        """Convenience interface to Python's "in" operator.

        Quickly tells if a given data type has getters and setters available
        """
        return data_type in self.typed_getter_setters

    def __getitem__(self, data_type: str) -> Tuple[Callable, Callable]:
        return self.typed_getter_setters[data_type]


class AttrCreationContext:
    """Context Manager: Used when creating and adding new dynamic attributes.

    AttrCreationContext objects have three jobs:
    1. Establish a short name if one isn't provided.
    2. Raise an error if the long or short name is already in use.
    3. Use the dependency graph modifier stack to add an attribute to the
       provided MirageNode.

    The attribute is not actually created by the context manager, it is only
    processed by it.  Calling functions must provide the underlying attribute
    object::

            with AttrCreationContext(*my_args) as attr_creator:
                attr_creator.m_attr = my_attr_maker_function()

    See any of the "add attr" methods on MirageNode to see examples of this
    pattern in use.
    """

    def __init__(
        self,
        mirage_node: MirageNode,
        long_name: str,
        short_name: Optional[str] = None,
        initial_value: Optional[Any] = None,
    ):
        """
        Args:
            mirage_node (MirageNode): The node that the new attribute will be
                assigned to.

            long_name (str): The long name of the attribute. By convention,
                this should be in camelCase.

            short_name (str): The short name of the attribute, typically no
                more than three characters.  If no short name is provided, the
                initials of the long name will be used.

            initial_value: The value that will be set right after creating the
                new attribute.  This is distinct from the default value.
        """
        self.mirage_node = mirage_node

        self.long_name = long_name

        self.short_name = short_name or get_short_attr_name(long_name)

        self._fail_on_existing_attr_name()

        self.initial_value = initial_value
        self.m_attr: OpenMaya.MObject
        self.mirage_attr: MirageAttr

    def _fail_on_existing_attr_name(self):
        """Raises an error if the long or short names are already in use."""
        for name in self.long_name, self.short_name:
            try:
                self.mirage_node[name]
            except Exception:
                pass
            else:
                raise AttributeError(
                    "The attribute name {name} is already in " "use".format(name=name)
                )

    def add_attr(self):
        """Adds the actual attribute object via the DG modifier stack."""
        with dg_modifier() as dg_mod:
            dg_mod.addAttribute(self.mirage_node._m_obj, self.m_attr)
        self.mirage_attr = self.mirage_node[self.long_name]
        if self.initial_value is not None:
            self.mirage_attr.value = self.initial_value

    def __enter__(self):
        """Returns this object for use in the calling scope."""
        return self

    def __exit__(self, *_):
        """Adds the attribute to the node if if has been created."""
        if hasattr(self, "m_attr") and self.m_attr is not None:
            self.add_attr()


# OPENMAYA ENUMERATOR UTILITIES


def _om_enum_to_dict(maya_enum) -> Dict[int, str]:
    """Converts an OpenMaya enumerator to a dict of {integer: string} pairs.

    Maya's "enumerators" are actually just classes with a bunch of static
    integer attributes. This function converts them to a dict so that they
    can be used more easily.
    """
    items = list(
        tuple(reversed(item))
        for item in maya_enum.__dict__.items()
        if item[0].startswith("k")
    )
    return dict(items)


def fn_type_string_to_int(type_as_string: str) -> int:
    """Returns the MFn.kWhatever integer if it exists"""

    # type_as_string is the ascii representation of the
    # maya object.  If it doesn't convert, returns
    # kInvalid, which maps to 0
    look_for = "".join(("k", type_as_string[0].upper(), type_as_string[1:]))

    try:
        return getattr(OpenMaya.MFn, look_for)
    except Exception:
        return OpenMaya.MFn.kInvalid


# MOBJECT UTILITIES


def m_obj_from_name(object_name: str) -> OpenMaya.MObject:
    """Gets an MObject from a maya string representation"""
    try:
        sel_list = OpenMaya.MGlobal.getSelectionListByName(object_name)
        return sel_list.getDependNode(0)
    except Exception:
        raise ValueError(
            "An exception was caught while trying to convert a string to an "
            "MObject.  The specified node ({name}) might not exist."
            "".format(name=object_name)
        )


def m_objs_from_names(object_name_iterable: Iterable[str]) -> List[OpenMaya.MObject]:
    """Gets a list of MObjects from an iterable of string representations."""
    sel_list = OpenMaya.MSelectionList()
    m_objs = list()
    for object_name in object_name_iterable:
        sel_list.add(object_name)
        m_objs.append(sel_list.getDependNode(0))
        sel_list.clear()
    return m_objs


def m_obj_from_uuid(object_uuid: str) -> OpenMaya.MObject:
    """Gets an MObject from a uuid as a string."""
    sel_list = OpenMaya.MSelectionList()
    muuid = OpenMaya.MUuid(object_uuid)
    sel_list.add(muuid)
    return sel_list.getDependNode(0)


# HIERARCHY UTILITIES


def _m_obj_children(m_obj: OpenMaya.MObject) -> List[OpenMaya.MObject]:
    """Gets the MObject children of an MObject."""
    fn_dag = _fn_dag_for_m_obj(m_obj)
    return [fn_dag.child(i) for i in range(fn_dag.childCount())]


def _m_obj_descendants(m_obj: OpenMaya.MObject) -> List[OpenMaya.MObject]:
    """Gets all of the descendants of the given MObject."""
    iterator = OpenMaya.MItDag()
    iterator.reset(m_obj)

    try:
        iterator.next()
    except RuntimeError:
        return list()

    all_descendants = list()
    while not iterator.isDone():
        all_descendants.append(iterator.currentItem())
        iterator.next()

    return all_descendants


def _m_obj_ancestors(
    m_obj: OpenMaya.MObject, node_filter: Iterable[OpenMaya.MObject] = []
):
    """Gets the ancestor MObjects of the given MObject"""
    # parent_of handles mobject conversion if necessary
    has_parents = True
    parent = m_obj
    parent_stack = []
    while has_parents:
        parent = _m_obj_parent(parent)
        if parent and parent not in node_filter:
            parent_stack.append(parent)
            has_parents = True
        else:
            has_parents = False
    return parent_stack


def _m_obj_parent(m_obj: OpenMaya.MObject) -> Optional[OpenMaya.MObject]:
    """Gets the parent MObject from the given MObject."""
    fn_dag = _fn_dag()
    fn_dag.setObject(m_obj)
    if fn_dag.parentCount() > 0:
        parent_m_obj = fn_dag.parent(0)
        return parent_m_obj
    else:
        return None


def get_world_node() -> MirageNode:
    """Returns the world node as a MirageNode"""
    iterator = OpenMaya.MItDag()
    return MirageNode.from_m_obj(iterator.root())


# CONNECTION UTILITIES


def make_connection(
    source_m_plug: OpenMaya.MPlug, dest_m_plug: OpenMaya.MPlug, force=True
):
    """Connects a source plug to a destination plug.

    Args:
        source_m_plug (OpenMaya.MPlug): The plug we want to connect from.
        dest_m_plug (OpenMaya.MPlug): The plug we want to connect to.
        force (bool): force the connection if one already exists on the
            destination plug.
    """
    with dg_modifier() as modifier:
        plug_connected = dest_m_plug.isDestination
        if force and plug_connected:
            break_source_connection(dest_m_plug)
        elif not force and plug_connected:
            raise RuntimeError(
                "The plug {} is already connected" "".format(dest_m_plug.name())
            )
        modifier.connect(source_m_plug, dest_m_plug)


def break_source_connection(m_plug: OpenMaya.MPlug):
    """Breaks the source connection (if any) on a given plug."""

    if m_plug.isDestination:
        with dg_modifier() as modifier:
            # this configuration of connectedTo always returns a
            # single plug in a list, so we'll just retrieve the plug
            source_m_plug = m_plug.connectedTo(True, False)[0]

            # copying the mplug prevents a maya crash
            source_m_plug = OpenMaya.MPlug(source_m_plug)

            modifier.disconnect(source_m_plug, m_plug)


# ATTRIBUTE TYPING UTILITIES


def get_attr_type(m_attribute: OpenMaya.MObject) -> str:
    """Gets the attribute type of the OpenMaya MObject object."""
    fn_set_list = OpenMaya.MGlobal.getFunctionSetList(m_attribute)
    try:
        attr_type = fn_set_list[2]
    except IndexError:
        raise AttributeError(
            "An error was raised attempting to get the "
            "attribute type for: {attr}".format(attr=m_attribute)
        )
    return attr_type


def get_data_type(m_attribute: OpenMaya.MObject, attr_type: str) -> str:
    """Gets the data type of the OpenMaya MObject.

    Not all attribute types have associated data types.

    Args:
        m_attribute (OpenMaya.MObject)
        attr_type (str): A valid Maya attribute type.
    """

    if attr_type == "kNumericAttribute":
        fn_numeric_attr = _fn_numeric_attr()
        fn_numeric_attr.setObject(m_attribute)
        return _numeric_type_dict()[fn_numeric_attr.numericType()]

    elif attr_type == "kTypedAttribute":
        fn_typed_attr = _fn_typed_attr()
        fn_typed_attr.setObject(m_attribute)
        return _data_type_dict()[fn_typed_attr.attrType()]

    elif attr_type == "kUnitAttribute":
        fn_unit_attr = _fn_unit_attr()
        fn_unit_attr.setObject(m_attribute)
        return _unit_type_dict()[fn_unit_attr.unitType()]

    else:
        return attr_type


# FUNCTION SET UTILITIES


def _fn_dag_for_m_obj(m_obj: OpenMaya.MObject) -> OpenMaya.MFnDagNode:
    """Given an MObject, return a DAG function set with the MObject assigned"""
    fn_dag = _fn_dag()
    fn_dag.setObject(m_obj)
    return fn_dag


def _fn_dg_for_m_obj(m_obj: OpenMaya.MObject) -> OpenMaya.MFnDependencyNode:
    """Returns a DG function set with the given MObject assigned."""
    fn_dg = _fn_dg()
    fn_dg.setObject(m_obj)
    return fn_dg


def _fn_set_for_m_obj(m_obj: OpenMaya.MObject) -> OpenMaya.MFnSet:
    """Returns a Set function set with the given MObject assigned."""
    fn_set = _fn_set()
    fn_set.setObject(m_obj)
    return fn_set


def _fn_attr_for_attribute(m_obj: OpenMaya.MObject) -> OpenMaya.MFnAttribute:
    """Returns a Attr function set with the given MObject assigned."""
    fn_attr = _fn_attr()
    fn_attr.setObject(m_obj)
    return fn_attr


def _fn_transform_for_m_obj(m_obj: OpenMaya.MObject) -> OpenMaya.MFnTransform:
    """Returns a Transform function set with the given MObject assigned."""
    fn_transform = _fn_transform()
    fn_transform.setObject(m_obj)
    return fn_transform


# UUID UTILITIES


def new_uuid() -> str:
    """Returns a new, unused uuid as a string"""
    new_uuid = OpenMaya.MUuid()
    new_uuid.generate()
    return new_uuid.asString()


# MODIFIER UTILITIES


@contextmanager
def dg_modifier():
    """Context Manager: Commit DG changes made while the context is open."""
    modifier = OpenMaya.MDGModifier()
    yield modifier
    modifier.doIt()


@contextmanager
def dag_modifier():
    """Context Manager: Commit DAG changes made while the context is open."""
    modifier = OpenMaya.MDagModifier()
    yield modifier
    modifier.doIt()


# STRING MANIPULATION UTILITIES


def camel_case_split(original_string) -> List[str]:
    """Split a string based on camelCase or CapWords.

    Original capitalization is kept in the returned strings.

    Args:
        original_string (str)

    Returns: list

    Example::

        camel_case_split("helloWorld")
        result:
            ['hello', 'World']
    """

    # the third-worst regex I've ever copied from the internet
    matches = re.finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", original_string
    )
    return [m.group(0) for m in matches]


def split_long_name(long_name: str) -> List[str]:
    """Splits a long name on its pipe characters.

    "|Group1|Group2|MyObject1" -> ["Group1", "Group2", "MyObject1"]
    "Lambert1" -> ["Lambert1"]
    """
    pipe = "|"
    if long_name.startswith(pipe):
        return long_name.split(pipe)[1:]
    return long_name.split(pipe)


def mel_attr_fmt(node_name: str, attr_name: str) -> str:
    """Formats a node name and attribute name into a maya-friendly string.

    "lambert", "message" -> "lambert.message"
    """
    return f"{node_name}.{attr_name}"


def get_short_attr_name(long_name: str) -> str:
    """Gets the initials from a long name in order to use as short name.

    Maya traditionally uses initials to represent short attribute names, so
    we'll do the same here.

    someLongName -> sln
    some_long_name -> sln
    someLong_name -> sln

    """
    underscore_split = long_name.split("_")
    tokens = list()
    for token in underscore_split:
        camel_split = camel_case_split(token)
        tokens.extend(camel_split)
    return "".join([token[0].lower() for token in tokens])
