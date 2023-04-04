"""
NOTE: this test suite must be run from a new, blank maya instance, with only
default plugins loaded in order to be valid.

To run the test suite in-situ, open maya's script editor and run:

    import unittest
    from mirage.tests import test_mirage as tst
    unittest.main(tst, exit=False)

Or, you can run this file directly from the command line:

    > mayapy "path/to/mirage/tests/test_mirage.py"

"""

import unittest
import mirage
from mirage import mirage as mirage_internal
from maya import cmds
from maya.api import OpenMaya

if not hasattr(cmds, "about"):
    import maya.standalone
    maya.standalone.initialize()

cmds.scriptEditorInfo(sr=False, se=True)  # only report errors

# UTILITIES

def cmds_attr(attr):
    return cmds.getAttr(attr.full_name)


def cmds_connected_attr(attr):
    return cmds.connectionInfo(attr.full_name, sourceFromDestination=True)


def cmds_compound_attr(attr):
    return cmds_attr(attr)[0]


# TEST CASES


class TestTestUtilities(unittest.TestCase):
    """Tests utilities for getting MObjects"""


class TestMObjectUtilities(unittest.TestCase):
    """Tests utilities for getting MObjects"""

    def test_string_to_mobj_returns_mobj(self):
        persp_cam_m_object = mirage.m_obj_from_name("persp")
        self.assertIsInstance(persp_cam_m_object, OpenMaya.MObject)

    def test_strings_to_mobj_returns_mobjs(self):
        cams = ["persp", "front", "top", "side"]
        m_objects = mirage.m_objs_from_names(cams)
        self.assertTrue(
            all(isinstance(m_object, OpenMaya.MObject) for m_object in m_objects)
        )
        self.assertEqual(len(m_objects), len(cams))


class TestDGModifierUtilities(unittest.TestCase):
    """Tests utilities for handling dependency-graph modifiers"""

    def setUp(self):
        self.shader_node_name = cmds.shadingNode("ramp", asTexture=True)
        self.shader_node_m_obj = mirage.m_obj_from_name(self.shader_node_name)
        self.shader_node_handle = OpenMaya.MObjectHandle(self.shader_node_m_obj)
        self.shader_dg_fn_set = OpenMaya.MFnDependencyNode(self.shader_node_m_obj)
        self.test_name = "new_name"

    def tearDown(self):
        for name in (self.shader_node_name, self.test_name):
            try:
                cmds.delete(name)
            except Exception:
                pass

    def test_dg_mod_stack(self):
        self.assertTrue(self.shader_node_handle.isValid())
        with mirage.dg_modifier() as modifier_stack:
            modifier_stack.renameNode(self.shader_node_m_obj, self.test_name)
        self.assertIsInstance(modifier_stack, OpenMaya.MDGModifier)
        self.assertEqual(self.shader_dg_fn_set.name(), self.test_name)


class TestDAGModifierUtilities(unittest.TestCase):
    """Tests utilities for Directed Acyclic Graph modifiers"""

    def setUp(self):
        self.cube_transform, cube_shape = cmds.polyCube()
        self.cube_node_m_obj = mirage.m_obj_from_name(self.cube_transform)
        self.cube_node_handle = OpenMaya.MObjectHandle(self.cube_node_m_obj)

    def tearDown(self):
        try:
            cmds.delete(self.cube_transform)
        except Exception:
            pass

    def test_dag(self):
        self.assertTrue(self.cube_node_handle.isValid())
        with mirage.dag_modifier() as modifier_stack:
            self.assertIsInstance(modifier_stack, OpenMaya.MDagModifier)
            modifier_stack.deleteNode(self.cube_node_m_obj)
        self.assertFalse(self.cube_node_handle.isValid())


class TestStringManipUtilities(unittest.TestCase):
    """Tests string manipulation utilities.

    Maya tracks most nodes using their names, which are unicode strings.
    """

    def test_split_long_name(self):
        long_name = "|Group1|Group2|MyObject1"
        expected_result = ["Group1", "Group2", "MyObject1"]
        split = mirage.split_long_name(long_name)
        self.assertCountEqual(split, expected_result)

    def test_split_long_name_no_pipes(self):
        long_name = "lambert1"
        expected_result = ["lambert1"]
        split = mirage.split_long_name(long_name)
        self.assertCountEqual(split, expected_result)

    def test_attr_fmt_func(self):
        node_name = "lambert1"
        attr_name = "outColor"
        expected_result = "lambert1.outColor"
        self.assertEqual(mirage.mel_attr_fmt(node_name, attr_name), expected_result)

    def test_attr_short_name_translator_camel_case_only(self):
        long_name = "someLongAttrName"
        expected_short_name = "slan"
        self.assertEqual(mirage.get_short_attr_name(long_name), expected_short_name)

    def test_attr_short_name_translator_underscores_only(self):
        long_name = "some_long_attr_name"
        expected_short_name = "slan"
        self.assertEqual(mirage.get_short_attr_name(long_name), expected_short_name)

    def test_attr_short_name_translator_mixed(self):
        long_name = "some_longAttr_Name"
        expected_short_name = "slan"
        self.assertEqual(mirage.get_short_attr_name(long_name), expected_short_name)


class TestMirageNodeCreation(unittest.TestCase):
    """Tests instantiation of MirageNodes and use of alternate constructors"""

    def test_mirage_node_cached_getter_name(self):
        node_one = mirage.MirageNode.from_name_cached("persp")
        node_two = mirage.MirageNode.from_name_cached("persp")
        self.assertIs(node_one, node_two)

    def test_mirage_node_cached_getter_m_obj(self):
        persp_cam_m_object = mirage.m_obj_from_name("persp")
        node_one = mirage.MirageNode.from_m_obj_cached(persp_cam_m_object)
        node_two = mirage.MirageNode.from_m_obj_cached(persp_cam_m_object)
        self.assertIs(node_one, node_two)

    def test_default_values(self):
        node = mirage.MirageNode()
        self.assertIsNone(node._hash_code)
        self.assertEqual(node._plugs, {})
        self.assertEqual(node._attrs, {})
        self.assertFalse(hasattr(node, "_m_obj"))

    def test_default_with_name(self):
        persp_cam_m_object = mirage.m_obj_from_name("persp")
        node = mirage.MirageNode("persp")
        self.assertIsInstance(node, mirage.MirageNode)
        self.assertEqual(persp_cam_m_object, node._m_obj)

    def test_from_m_object(self):
        persp_cam_m_object = mirage.m_obj_from_name("persp")
        node = mirage.MirageNode.from_m_obj(persp_cam_m_object)
        self.assertIsInstance(node, mirage.MirageNode)
        self.assertEqual(persp_cam_m_object, node._m_obj)

    def test_from_name(self):
        node = mirage.MirageNode.from_name("persp")
        self.assertIsInstance(node, mirage.MirageNode)
        self.assertIsNotNone(node._m_obj)

    def test_from_names(self):
        cams = ["persp", "front", "top", "side"]
        nodes = mirage.MirageNode.from_names(cams)
        self.assertTrue(all(isinstance(node, mirage.MirageNode) for node in nodes))

    def test_from_uuid(self):
        persp_cam_m_obj = mirage.m_obj_from_name("persp")
        uuid = OpenMaya.MFnDependencyNode(persp_cam_m_obj).uuid()
        node = mirage.MirageNode.from_uuid(str(uuid))
        self.assertEqual(persp_cam_m_obj, node._m_obj)

    def test_from_uuid_cached(self):
        persp_cam = mirage.MirageNode.from_name_cached("persp")
        node = mirage.MirageNode.from_uuid_cached(persp_cam.uuid)
        self.assertIs(persp_cam, node)


class TestMirageNodeCmdCreation(unittest.TestCase):
    """Tests instantiation of MirageNodes from cmds-module-based commands"""

    def tearDown(self):
        cmds.delete("test_cube*")

    def test_from_cmd(self):
        nodes = mirage.MirageNode.from_cmd("polyCube", name="test_cube")
        self.assertTrue(bool(cmds.ls("test_cube*")))
        self.assertIsInstance(nodes, list)
        self.assertTrue(all(isinstance(n, mirage.MirageNode) for n in nodes))


class TestMirageNodeInstanceAttrs(unittest.TestCase):
    """Tests the various lazily-evaluated and actively-evaluated properties
    of MirageNodes, as well as magic attributes and property assignment"""

    def setUp(self):
        self.cube_node_name = cmds.polyCube(name="test_cube")[0]
        self.cube_node = mirage.MirageNode(self.cube_node_name)
        self.lambert_node_name = "lambert1"
        self.lambert_node = mirage.MirageNode.from_name(self.lambert_node_name)
        self.persp_cam_node_name = "persp"
        self.persp_cam_node = mirage.MirageNode.from_name(self.persp_cam_node_name)

    def tearDown(self):
        try:
            cmds.delete("test_cube*")
        except ValueError:
            pass

    def test_fn_dg_property(self):
        self.assertIsInstance(self.cube_node.fn_dg, OpenMaya.MFnDependencyNode)
        self.assertEqual(self.cube_node.fn_dg.object(), self.cube_node._m_obj)

    def test_fn_dag_property(self):
        self.assertIsInstance(self.cube_node.fn_dag, OpenMaya.MFnDagNode)
        self.assertEqual(self.cube_node.fn_dag.object(), self.cube_node._m_obj)

    def test_fn_dag_fails_on_dg_node(self):
        with self.assertRaises(RuntimeError):
            self.lambert_node.fn_dag

    def test_name_property(self):
        self.assertEqual(self.cube_node.name, self.cube_node_name)

    def test_long_name_property(self):
        expected_result = f"|{self.cube_node_name}"
        self.assertEqual(self.cube_node.long_name, expected_result)

    def test_long_name_works_on_dg_node(self):
        self.assertEqual(self.lambert_node.long_name, self.lambert_node_name)

    def test_short_name_property(self):
        self.assertEqual(self.cube_node.short_name, self.cube_node_name)

    def test_short_name_works_on_dg_node(self):
        self.assertEqual(self.lambert_node.short_name, self.lambert_node_name)

    def test_name_assign(self):
        new_name = "test_cube_renamed"
        self.assertNotEqual(self.cube_node.name, new_name)
        self.cube_node.name = new_name
        self.assertEqual(self.cube_node.name, new_name)

    def test_uuid_property(self):
        cube_m_obj = mirage.m_obj_from_name(self.cube_node_name)
        uuid = OpenMaya.MFnDependencyNode(cube_m_obj).uuid().asString()
        self.assertEqual(self.cube_node.uuid, uuid)

    def test_uuid_assign(self):
        new_uuid = mirage.new_uuid()
        self.assertNotEqual(self.cube_node.uuid, new_uuid)
        self.cube_node.uuid = new_uuid
        self.assertEqual(self.cube_node.uuid, new_uuid)
        self.assertEqual(self.cube_node.fn_dg.uuid().asString(), new_uuid)

    def test_is_dag(self):
        self.assertTrue(self.cube_node.is_dag)
        self.assertFalse(self.lambert_node.is_dag)

    def test_is_valid(self):
        self.assertTrue(self.cube_node.valid)
        cmds.delete(self.cube_node.name)
        self.assertFalse(self.cube_node.valid)

    def test_is_default(self):
        self.assertTrue(self.lambert_node.is_default)
        self.assertFalse(self.cube_node.is_default)

    def test_type_name(self):
        lambert_type_name = "lambert"
        self.assertEqual(self.lambert_node.type_name, lambert_type_name)
        cube_type_name = "transform"
        self.assertEqual(self.cube_node.type_name, cube_type_name)

    def test_api_type(self):
        lambert_api_type = "kLambert"
        self.assertEqual(self.lambert_node.api_type, lambert_api_type)
        cube_api_type = "kTransform"
        self.assertEqual(self.cube_node.api_type, cube_api_type)

    def test_classification_str(self):
        lambert_classification = "drawdb/shader/surface/lambert:shader/surface"
        self.assertEqual(self.lambert_node.classification, lambert_classification)
        cube_classification = "drawdb/geometry/transform"
        self.assertEqual(self.cube_node.classification, cube_classification)

    def test_classifications(self):
        expectation = ["drawdb", "shader", "surface", "lambert:shader", "surface"]
        self.assertEqual(self.lambert_node.classifications, expectation)

    def test_inherited_classifications(self):
        expectation = ["containerBase", "entity", "dagNode", "transform"]
        self.assertEqual(self.cube_node.inherited_types, expectation)

    def test_selection_state(self):
        cmds.select(clear=True)
        self.assertFalse(self.cube_node.selected)
        cmds.select(self.cube_node.name)
        self.assertTrue(self.cube_node.selected)

    def test_locked_property(self):
        self.assertFalse(self.cube_node.locked)
        self.cube_node.locked = True
        self.assertTrue(self.cube_node.locked)
        self.cube_node.locked = False

    def test_equality_manual_setup(self):
        node_one = mirage.MirageNode()
        node_two = mirage.MirageNode()
        m_object = mirage.m_obj_from_name("persp")
        self.assertNotEqual(node_one, node_two)
        node_one._m_obj = m_object
        node_two._m_obj = m_object
        self.assertEqual(node_one, node_two)
        self.assertNotEqual(self.persp_cam_node, self.cube_node)

    def test_equality_using_constructor(self):
        node_one = mirage.MirageNode("persp")
        node_two = mirage.MirageNode("persp")
        self.assertEqual(node_one, node_two)
        self.assertIsNot(node_one, node_two)

    def test_get_m_dag_path(self):
        self.assertIsInstance(self.persp_cam_node._m_dag_path, OpenMaya.MDagPath)

    def test_m_dag_path_raises_typeerror(self):
        with self.assertRaises(TypeError):
            self.lambert_node._m_dag_path

    def test_shape_property_raises_typeerror(self):
        with self.assertRaises(TypeError):
            self.lambert_node.shape

    def test_deletion(self):
        self.assertTrue(self.cube_node.valid)
        self.cube_node.delete()
        self.assertFalse(self.cube_node.valid)


class TestMirageNodeHierarchyUtils(unittest.TestCase):
    """Tests the various tools for querying and editing DAG hierarchies"""

    def setUp(self):
        self.cube_node_name = cmds.polyCube(name="test_cube")[0]
        self.cube_node = mirage.MirageNode(self.cube_node_name)
        self.sphere_node_name = cmds.polySphere(name="test_sphere")[0]
        self.sphere_node = mirage.MirageNode(self.sphere_node_name)
        self.persp_cam_node = mirage.MirageNode("persp")

    def tearDown(self):
        for node_name in (self.cube_node_name, self.sphere_node_name):
            try:
                cmds.delete(node_name)
            except Exception:
                pass

    def test_parent_property(self):
        assert self.cube_node.parent
        self.assertEqual(self.cube_node.parent.name, "world")
        self.cube_node.parent = self.sphere_node
        self.assertEqual(self.cube_node.parent, self.sphere_node)

    def test_parent_to_world(self):
        self.cube_node.parent = self.sphere_node
        self.assertEqual(self.cube_node.parent, self.sphere_node)
        self.cube_node.parent = None
        assert self.cube_node.parent
        self.assertEqual(self.cube_node.parent.name, "world")

    def test_adoption(self):
        self.cube_node.adopt(self.sphere_node)
        self.assertEqual(self.sphere_node.parent, self.cube_node)

    def test_children(self):
        self.cube_node.adopt(self.sphere_node)
        self.assertIn(self.sphere_node, self.cube_node.children)

        # there should be two children, the sphere transform and the cube's
        # shape node
        self.assertEqual(len(self.cube_node.children), 2)

    def test_descendants(self):
        self.cube_node.adopt(self.sphere_node)
        self.assertIn(self.sphere_node, self.cube_node.descendants)

        # there should be three descendants, the sphere transform, sphere's
        # shape node and the cube's shape node
        self.assertEqual(len(self.cube_node.descendants), 3)

    def test_ancestors(self):
        self.persp_cam_node.adopt(self.sphere_node)
        self.sphere_node.adopt(self.cube_node)
        cube_ancestors = self.cube_node.ancestors
        self.assertIn(self.sphere_node, cube_ancestors)
        self.assertIn(self.persp_cam_node, cube_ancestors)
        self.assertEqual(len(cube_ancestors), 3)

    def test_shape_node(self):
        self.assertEqual(self.sphere_node.children[0], self.sphere_node.shape)


class TestMirageNodeAttributeLookup(unittest.TestCase):
    """Tests getting plugs and MirageAttrs from MirageNodes"""

    def setUp(self):
        self.lambert_node_name = "lambert1"
        self.lambert_node = mirage.MirageNode.from_name(self.lambert_node_name)
        self.persp_node_name = "persp"
        self.persp_node = mirage.MirageNode.from_name(self.persp_node_name)

    def test_plug_lookup_return_type(self):
        self.assertIsInstance(self.lambert_node.plug("color"), OpenMaya.MPlug)

    def test_plug_error_raises_attributeerror(self):
        with self.assertRaises(AttributeError):
            self.lambert_node.plug("transform")

    def test_get_all_plugs(self):
        # lambert has 52 top-level attribute plugs
        self.assertEqual(len(self.lambert_node.all_plugs), 52)
        self.assertIn("color", self.lambert_node._plugs)

    def test_get_all_connected_plugs(self):
        lightset_node = mirage.MirageNode.from_name("defaultLightSet")
        self.assertEqual(len(lightset_node.all_connected_plugs), 1)
        self.assertEqual(
            lightset_node.all_connected_plugs[0].name(), "defaultLightSet.message"
        )
        self.assertIn("message", lightset_node._plugs)

    def test_list_attributes_dg_node(self):
        self.assertEqual(
            len(self.lambert_node.list_attributes()), len(self.lambert_node.all_plugs)
        )

    def test_list_attributes_dg_node_graceful_failure_on_extend_to_shape(self):
        self.assertEqual(
            len(self.lambert_node.list_attributes(extend_to_shape=True)),
            len(self.lambert_node.all_plugs),
        )

    def test_list_attributes_dag_node_no_shape_extend(self):
        self.assertEqual(
            len(self.persp_node.list_attributes(extend_to_shape=False)),
            len(self.persp_node.all_plugs),
        )

    def test_list_attributes_dag_node_shape_extend(self):
        self.assertEqual(
            len(self.persp_node.list_attributes(extend_to_shape=True)),
            (len(self.persp_node.all_plugs) + len(self.persp_node.shape.all_plugs)),
        )

    def test_get_api_attribute_function(self):
        msg = self.lambert_node.attr("message")
        self.assertIsInstance(msg, mirage.MirageAttr)
        self.assertIs(msg.mirage_node, self.lambert_node)
        self.assertIs(msg._m_plug, self.lambert_node.plug("message"))

    def test_get_nonexistant_api_attr_raises_attributeerror(self):
        with self.assertRaises(AttributeError):
            self.lambert_node.attr("transform")


class TestMirageAttrCreation(unittest.TestCase):
    """Tests instantiation of MirageAttrs"""

    def setUp(self):
        self.persp_cam_node = mirage.MirageNode.from_name_cached("persp")
        self.plug = self.persp_cam_node.plug("message")

    def test_api_attr_instantiation_with_plug_obj(self):
        api_attr = mirage.MirageAttr(plug=self.plug)
        self.assertIs(api_attr._m_plug, self.plug)

    def test_api_attr_from_name(self):
        api_attr = mirage.MirageAttr.from_full_name("persp.message")
        self.assertIs(api_attr.mirage_node, self.persp_cam_node)
        self.assertIs(api_attr._m_plug, self.plug)

    def test_api_attr_from_node_and_name(self):
        api_attr = mirage.MirageAttr.from_mirage_node_and_name(
            self.persp_cam_node, "message"
        )
        self.assertIs(api_attr.mirage_node, self.persp_cam_node)
        self.assertIs(api_attr._m_plug, self.plug)

    def test_api_attr_instantiation_from_getitem_on_mirage_node(self):
        api_attr = self.persp_cam_node["translate"]
        self.assertIsInstance(api_attr, mirage.MirageAttr)


class TestMirageAttrProperties(unittest.TestCase):
    """Tests the lazily- and actively-evaluated properties of MirageAttrs"""

    def setUp(self):
        self.persp_cam_node = mirage.MirageNode.from_name_cached("persp")
        self.msg_attr = self.persp_cam_node.attr("message")
        self.translate_attr = self.persp_cam_node.attr("translate")
        self.matrix_attr = self.persp_cam_node.attr("worldMatrix")
        surface = "initialShadingGroup.surfaceShader"
        self.lambert_connection = mirage.MirageAttr.from_full_name(surface)

    def test_attribute(self):
        self.assertIsInstance(self.msg_attr.attribute, OpenMaya.MObject)

    def test_full_name(self):
        self.assertEqual(self.msg_attr.full_name, "persp.message")
        self.assertEqual(self.translate_attr.full_name, "persp.translate")
        self.assertEqual(self.matrix_attr.full_name, "persp.worldMatrix")

    def test_name(self):
        self.assertEqual(self.msg_attr.name, "message")
        self.assertEqual(self.translate_attr.name, "translate")
        self.assertEqual(self.matrix_attr.name, "worldMatrix")

    def test_attr_type(self):
        self.assertEqual(self.msg_attr.attr_type, "kMessageAttribute")
        self.assertEqual(self.translate_attr.attr_type, "kNumericAttribute")
        self.assertEqual(self.matrix_attr.attr_type, "kTypedAttribute")

    def test_is_array(self):
        self.assertFalse(self.msg_attr.is_array)
        self.assertFalse(self.translate_attr.is_array)
        self.assertTrue(self.matrix_attr.is_array)

    def test_is_compound(self):
        self.assertFalse(self.msg_attr.is_compound)
        self.assertTrue(self.translate_attr.is_compound)
        self.assertTrue(self.matrix_attr.is_array)

    def test_is_destination_connection(self):
        self.assertTrue(self.lambert_connection.is_destination)
        self.assertFalse(self.translate_attr.is_destination)

    def test_data_type(self):
        self.assertEqual(self.msg_attr.data_type, "kMessageAttribute")
        self.assertEqual(self.translate_attr.data_type, "k3Double")
        self.assertEqual(self.matrix_attr.data_type, "kMatrix")


class TestGetMirageAttrSimpleValues(unittest.TestCase):
    """Tests getting simple (non-compound, non-array) attributes"""

    def setUp(self):
        self.time_node = mirage.MirageNode.from_name("time1")
        self.lambert_node = mirage.MirageNode.from_name("lambert1")
        self.persp_node = mirage.MirageNode.from_name("persp")

    # NUMERIC DATA TYPES

    def test_get_boolean(self):
        attr = self.time_node["caching"]
        result = attr.value
        self.assertFalse(result)
        self.assertEqual(cmds_attr(attr), result)

    def test_get_byte(self):
        attr = self.time_node["isHistoricallyInteresting"]
        result = attr.value
        self.assertEqual(result, 2)
        self.assertEqual(cmds_attr(attr), result)

    def test_get_long(self):
        # none of the default maya nodes have a kLong-type attribute
        pass

    def test_get_int(self):
        attr = self.lambert_node["primitiveId"]
        result = attr.value
        self.assertEqual(result, 0)
        self.assertEqual(cmds_attr(attr), result)

    def test_get_int_64(self):
        # none of the default maya nodes have an int64-type attribute
        pass

    def test_get_short(self):
        attr = self.lambert_node["refractionLimit"]
        result = attr.value
        self.assertEqual(result, 6)
        self.assertEqual(cmds_attr(attr), result)

    def test_get_enum_index(self):
        attr = self.time_node["nodeState"]
        result = attr.value
        self.assertEqual(result, 0)
        self.assertEqual(cmds_attr(attr), result)

    def test_get_float(self):
        attr = self.lambert_node["diffuse"]
        result = attr.value
        self.assertAlmostEqual(result, 0.8)
        self.assertAlmostEqual(cmds_attr(attr), result)

    def test_get_unit(self):
        # none of the default maya nodes have a kUnitAttribute-type attribute
        pass

    def test_get_time(self):
        attr = self.time_node["outTime"]
        result = attr.value
        self.assertAlmostEqual(result, 1.0)
        self.assertAlmostEqual(cmds_attr(attr), result)

    def test_get_double(self):
        attr = self.lambert_node["vrEdgeWeight"]
        result = attr.value
        self.assertAlmostEqual(result, 0.0)
        self.assertAlmostEqual(cmds_attr(attr), result)

    def test_get_addr(self):
        attr = self.lambert_node["objectId"]
        result = attr.value
        self.assertAlmostEqual(result, 0.0)
        self.assertAlmostEqual(cmds_attr(attr), result)

    def test_get_distance(self):
        attr = self.persp_node["scalePivotX"]
        result = attr.value
        self.assertAlmostEqual(result, 0.0)
        self.assertAlmostEqual(cmds_attr(attr), result)

    def test_get_angle(self):
        attr = self.persp_node["shutterAngle"]
        result = attr.value
        self.assertAlmostEqual(result, 144.0)
        self.assertAlmostEqual(cmds_attr(attr), result)

    # STRINGS

    def test_get_string(self):
        attr = self.persp_node["imageName"]
        result = attr.value
        self.assertEqual(result, "persp")
        self.assertEqual(cmds_attr(attr), result)

    def test_get_char(self):
        # none of the default maya nodes have a Char-type attribute
        pass


class TestMirageAttrNoneReturnAttributes(unittest.TestCase):
    """Tests getting Attributes without well-defined data types"""

    def setUp(self):
        self.time_node = mirage.MirageNode.from_name("time1")
        self.lambert_node = mirage.MirageNode.from_name("lambert1")
        self.persp_node = mirage.MirageNode.from_name("persp")

    def test_get_invalid(self):
        attr = self.time_node["timewarpIn_Hidden"]
        result = attr.value
        self.assertEqual(result, None)
        self.assertEqual(cmds_attr(attr), result)

    def test_get_any(self):
        attr = self.persp_node["specifiedManipLocation"]
        result = attr.value
        self.assertEqual(result, None)
        self.assertEqual(cmds_attr(attr), result)

    def test_get_generic(self):
        attr = self.persp_node["geometry"]
        result = attr.value
        self.assertEqual(result, None)
        self.assertEqual(cmds_attr(attr), result)


class TestMirageAttrSimpleValueAssignment(unittest.TestCase):
    """Tests setting simple (non-compound, non-array) attributes"""

    def setUp(self):
        self.time_node = mirage.MirageNode.from_name("time1")
        self.lambert_node = mirage.MirageNode.from_name("lambert1")
        self.persp_node = mirage.MirageNode.from_name("persp")
        self.cube_node, _ = mirage.MirageNode.from_cmd("polyCube", name="test_cube")
        self.translate_x = self.cube_node["translateX"]
        self.scale_x = self.cube_node["scaleX"]
        self.visibility = self.cube_node["visibility"]

    def tearDown(self):
        try:
            cmds.delete(self.cube_node.name)
        except Exception:
            pass

    def test_get_numeric_values_after_changed_via_cmds(self):
        move_to = 5.0
        scale_to = 2.0
        cmds.setAttr(self.translate_x.full_name, move_to)
        cmds.setAttr(self.scale_x.full_name, scale_to)
        self.assertAlmostEqual(self.translate_x.value, move_to)
        self.assertAlmostEqual(self.scale_x.value, scale_to)

    def test_get_numeric_values_set_via_api_attr_interface(self):
        move_to = 2.0
        scale_to = 3.0
        self.cube_node["translateX"] = move_to
        self.cube_node["scaleX"] = scale_to
        self.assertAlmostEqual(self.translate_x.value, move_to)
        self.assertAlmostEqual(self.scale_x.value, scale_to)

    def test_get_numeric_values_set_via_mirage_node_interface(self):
        move_to = 2.0
        scale_to = 3.0
        self.translate_x.value = move_to
        self.scale_x.value = scale_to
        self.assertAlmostEqual(self.translate_x.value, move_to)
        self.assertAlmostEqual(self.scale_x.value, scale_to)

    def test_get_numeric_values_match_cmds_getattr(self):
        move_to = 7.0
        scale_to = 1.5
        self.translate_x.value = move_to
        self.scale_x.value = scale_to
        self.assertAlmostEqual(
            self.translate_x.value, cmds.getAttr(self.translate_x.full_name)
        )
        self.assertAlmostEqual(self.scale_x.value, cmds.getAttr(self.scale_x.full_name))

    def test_set_bool(self):
        self.visibility.value = False
        self.assertFalse(self.visibility.value)
        self.assertEqual(self.visibility.value, cmds.getAttr(self.visibility.full_name))

    def test_compound_assignment_add(self):
        self.translate_x.value = 2.0
        self.translate_x.value += 3.0
        self.assertAlmostEqual(self.translate_x.value, 5.0)

    def test_compound_assignment_sub(self):
        self.translate_x.value = 2.0
        self.translate_x.value -= 3.0
        self.assertAlmostEqual(self.translate_x.value, -1.0)

    def test_compound_assignment_mul(self):
        self.translate_x.value = 2.0
        self.translate_x.value *= 3.0
        self.assertAlmostEqual(self.translate_x.value, 6.0)


class TestGetApiMultiAttributes(unittest.TestCase):
    """Tests getting "multi" (compound, array) attributes"""

    def setUp(self):
        self.lambert_node = mirage.MirageNode.from_name("lambert1")
        self.persp_node = mirage.MirageNode.from_name("persp")
        self.default_render_layer_node = mirage.MirageNode.from_name(
            "defaultRenderLayer"
        )

    def test_get_k2Int(self):
        # none of the default maya nodes have a k2Int-type attribute
        pass

    def test_get_k3Int(self):
        # none of the default maya nodes have a k3Int-type attribute
        pass

    def test_get_k2Float(self):
        attr = self.default_render_layer_node["outSize"]
        self.assertCountEqual(attr.value, [0.0, 0.0])
        self.assertCountEqual(attr.value, cmds_compound_attr(attr))

    def test_get_k3Float(self):
        attr = self.lambert_node["color"]
        self.assertCountEqual(attr.value, [0.5, 0.5, 0.5])
        self.assertCountEqual(attr.value, cmds_compound_attr(attr))

    def test_get_k2Double(self):
        attr = self.persp_node["shake"]
        self.assertCountEqual(attr.value, [0.0, 0.0])
        self.assertCountEqual(attr.value, cmds_compound_attr(attr))

    def test_get_k3Double(self):
        attr = self.persp_node["selectHandle"]
        self.assertCountEqual(attr.value, [0.0, 0.0, 0.0])
        self.assertCountEqual(attr.value, cmds_compound_attr(attr))

    def test_get_k4Double(self):
        attr = self.persp_node["rotateQuaternion"]
        self.assertCountEqual(attr.value, [0.0, 0.0, 0.0, 0.0])
        self.assertCountEqual(attr.value, cmds_compound_attr(attr))

    def test_get_kCompoundAttr(self):
        attr = self.persp_node["renderInfo"]
        self.assertCountEqual(attr.value, [0, True, 0])
        self.assertCountEqual(attr.value, cmds_compound_attr(attr))

    def test_get_kMatrix(self):
        attr = self.persp_node["parentMatrix"]
        self.assertCountEqual(
            attr.value,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )


class TestGetMirageAttrConnections(unittest.TestCase):
    """Tests retrieval of attribute connections via MirageAttr"""

    def setUp(self):
        # NOTE: this test will fail in Maya below 2024
        # TODO: find a better default node to run this test against, or make
        # a new node to test against
        self.surface_node = mirage.MirageNode("standardSurface1")
        self.initial_shading_group = mirage.MirageNode("initialShadingGroup")
        self.light_link_node = mirage.MirageNode("lightLinker1")

    def test_get_destination_connection(self):
        self.assertEqual(
            self.initial_shading_group["surfaceShader"].value,
            self.surface_node["outColor"],
        )

    def test_get_nested_compound_connection(self):
        attr = self.light_link_node["link"]
        self.assertTrue(
            all(isinstance(value[0], mirage.MirageAttr) for value in attr.value)
        )
        with self.assertRaises(RuntimeError):
            cmds_compound_attr(attr)


class TestSetMultiAttributesDAG(unittest.TestCase):
    """Tests setting multi (compound / array) attributes on DAG nodes."""

    def setUp(self):
        self.cube_node, _ = mirage.MirageNode.from_cmd("polyCube", name="test_cube")

    def tearDown(self):
        try:
            cmds.delete(self.cube_node.name)
        except Exception:
            pass

    def test_set_typed_compound_attribute(self):
        expected_default_value = [0, 0, 0]
        new_value = [1, 2, 3]

        self.assertEqual(self.cube_node["translate"].value, expected_default_value)
        self.cube_node["translate"] = new_value
        self.assertEqual(self.cube_node["translate"].value, new_value)

    def test_compound_assignment_compound_operand_add(self):
        expected_default_value = [0, 0, 0]
        new_value = [1, 2, 3]
        third_value = [2, 3, 4]

        self.assertEqual(self.cube_node["translate"].value, expected_default_value)
        self.cube_node["translate"] += new_value
        self.assertEqual(self.cube_node["translate"].value, new_value)
        self.cube_node["translate"] += [1, 1, 1]
        self.assertEqual(self.cube_node["translate"].value, third_value)

    def test_compound_assignment_compound_operand_sub(self):
        expected_default_value = [0, 0, 0]
        new_value = [-1, -2, -3]
        third_value = [-2, -3, -4]

        self.assertEqual(self.cube_node["translate"].value, expected_default_value)
        self.cube_node["translate"] -= [1, 2, 3]
        self.assertEqual(self.cube_node["translate"].value, new_value)
        self.cube_node["translate"] -= [1, 1, 1]
        self.assertEqual(self.cube_node["translate"].value, third_value)

    def test_compound_assignment_compound_operand_mul(self):
        expected_default_value = [0, 0, 0]
        new_value = [1, 2, 3]
        third_value = [2, 4, 6]

        self.assertEqual(self.cube_node["translate"].value, expected_default_value)
        self.cube_node["translate"] += [1, 1, 1]
        self.cube_node["translate"] *= [1, 2, 3]
        self.assertEqual(self.cube_node["translate"].value, new_value)
        self.cube_node["translate"] *= [2, 2, 2]
        self.assertEqual(self.cube_node["translate"].value, third_value)

    def test_set_matrix_array_value(self):
        # All of the default maya matrix attributes aren't writable, so we
        # need to make a new one
        attr = OpenMaya.MFnMatrixAttribute()
        node_attr = attr.create("myMatrix", "mm")
        attr.readable = True
        attr.writable = True
        attr.storable = True
        self.cube_node.fn_dg.addAttribute(node_attr)

        expected_default_value = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        new_value = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]

        self.assertEqual(self.cube_node["myMatrix"].value, expected_default_value)
        self.assertEqual(self.cube_node["translateY"].value, 0.0)
        self.cube_node["myMatrix"] = new_value
        self.assertEqual(self.cube_node["myMatrix"].value, new_value)


class TestSetMultiAttributesDG(unittest.TestCase):
    """Tests setting multi (compound / array) attributes on DG nodes."""

    def setUp(self):
        self.ramp = mirage.MirageNode.from_cmd(
            "shadingNode", "ramp", asTexture=True, name="test_ramp"
        )[0]

    def tearDown(self):
        try:
            cmds.delete(self.ramp.name)
        except Exception:
            pass

    def test_set_array_on_index(self):
        new_value = [0.0, [1, 0, 0]]
        self.ramp["colorEntryList"][0] = new_value
        self.assertCountEqual(self.ramp["colorEntryList"][0].value, new_value)

    def test_set_array_implied_indices(self):
        new_value = [[0.0, [1, 0, 0]], [1.0, [1, 1, 0]]]
        self.ramp["colorEntryList"] = new_value
        self.assertCountEqual(self.ramp["colorEntryList"].value, new_value)


class TestAttrCreationUtils(unittest.TestCase):
    def setUp(self):
        self.cube_node, _ = mirage.MirageNode.from_cmd("polyCube", name="test_cube")

    def tearDown(self):
        try:
            cmds.delete(self.cube_node.name)
        except Exception:
            pass

    def test_context_manager_errors_on_existing_attr(self):
        with self.assertRaises(AttributeError):
            mirage_internal.AttrCreationContext(self.cube_node, "translate", "t")

    def test_context_manager_graceful_exit_when_no_attr_created(self):
        with mirage_internal.AttrCreationContext(self.cube_node, "someNewName"):
            pass

    def test_create_numeric_attr_no_default(self):
        with mirage_internal.AttrCreationContext(self.cube_node, "someNewName") as acc:
            acc.m_attr = mirage_internal.make_numeric_attr(
                "someNewName", "snn", OpenMaya.MFnNumericData.kDouble  # type: ignore
            )
        self.assertAlmostEqual(self.cube_node["someNewName"].value, 0.0)

    def test_create_numeric_attr_with_default(self):
        default_value = 5.0
        with mirage_internal.AttrCreationContext(self.cube_node, "someNewName") as acc:
            acc.m_attr = mirage_internal.make_numeric_attr(
                "someNewName",
                "snn",
                OpenMaya.MFnNumericData.kDouble,  # type: ignore
                default_value=default_value,
            )
        self.assertAlmostEqual(self.cube_node["someNewName"].value, default_value)


class TestMirageNodeAddAttributeMethods(unittest.TestCase):
    def setUp(self):
        self.cube_node, _ = mirage.MirageNode.from_cmd("polyCube", name="test_cube")

    def tearDown(self):
        try:
            cmds.delete(self.cube_node.name)
        except Exception:
            pass

    def test_add_int_attr(self):
        expected_result = 0
        new_attr = self.cube_node.add_int_attr("newIntAttr")
        self.assertEqual(new_attr.value, expected_result)

    def test_add_int_attr_default(self):
        default = 1
        new_attr = self.cube_node.add_int_attr("newIntAttr", default_value=default)
        self.assertEqual(new_attr.value, default)
        self.assertTrue(new_attr.is_default_value)

    def test_add_int_attr_initial_value(self):
        initial = 2
        new_attr = self.cube_node.add_int_attr("newIntAttr", initial_value=initial)
        self.assertEqual(new_attr.value, initial)

    def test_add_int_attr_initial_overrides_default_value(self):
        default = 1
        initial = 2
        new_attr = self.cube_node.add_int_attr(
            "newIntAttr", default_value=default, initial_value=initial
        )
        self.assertEqual(new_attr.value, initial)
        self.assertFalse(new_attr.is_default_value)

    def test_add_float_attr(self):
        expected_result = 0.0
        new_attr = self.cube_node.add_float_attr("newFloatAttr")
        self.assertAlmostEqual(new_attr.value, expected_result)

    def test_add_float_attr_default(self):
        default = 1.0
        new_attr = self.cube_node.add_float_attr("newFloatAttr", default_value=default)
        self.assertAlmostEqual(new_attr.value, default)
        self.assertTrue(new_attr.is_default_value)

    def test_add_float_attr_initial_value(self):
        initial = 2.0
        new_attr = self.cube_node.add_float_attr("newFloatAttr", initial_value=initial)
        self.assertAlmostEqual(new_attr.value, initial)

    def test_add_float_attr_initial_overrides_default_value(self):
        default = 1.0
        initial = 2.0
        new_attr = self.cube_node.add_float_attr(
            "newFloatAttr", default_value=default, initial_value=initial
        )
        self.assertAlmostEqual(new_attr.value, initial)
        self.assertFalse(new_attr.is_default_value)

    def test_add_bool_attr(self):
        expected_result = False
        new_attr = self.cube_node.add_bool_attr("newBoolAttr")
        self.assertIs(new_attr.value, expected_result)

    def test_add_bool_attr_default(self):
        default = True
        new_attr = self.cube_node.add_bool_attr("newBoolAttr", default_value=default)
        self.assertIs(new_attr.value, default)
        self.assertTrue(new_attr.is_default_value)

    def test_add_bool_attr_initial_value(self):
        initial = True
        new_attr = self.cube_node.add_bool_attr("newBoolAttr", initial_value=initial)
        self.assertIs(new_attr.value, initial)

    def test_add_bool_attr_initial_overrides_default_value(self):
        default = False
        initial = True
        new_attr = self.cube_node.add_bool_attr(
            "newBoolAttr", default_value=default, initial_value=initial
        )
        self.assertIs(new_attr.value, initial)
        self.assertFalse(new_attr.is_default_value)

    def test_add_string_attr(self):
        expected_result = ""
        new_attr = self.cube_node.add_string_attr("newStringAttr")
        self.assertEqual(new_attr.value, expected_result)

    # TODO: This test fails in Maya 2024; no bug report seems to match
    def test_add_string_attr_default(self):
        default = "default"
        new_attr = self.cube_node.add_string_attr(
            "newStringAttr", default_value=default
        )
        self.assertEqual(new_attr.value, default)
        self.assertTrue(new_attr.is_default_value)

    def test_add_string_attr_initial_value(self):
        initial = "initial"
        new_attr = self.cube_node.add_string_attr(
            "newStringAttr", initial_value=initial
        )
        self.assertEqual(new_attr.value, initial)

    def test_add_string_attr_initial_overrides_default_value(self):
        default = "default"
        initial = "initial"
        new_attr = self.cube_node.add_string_attr(
            "newStringAttr", default_value=default, initial_value=initial
        )
        self.assertEqual(new_attr.value, initial)
        self.assertFalse(new_attr.is_default_value)

    def test_add_vector_attr(self):
        expected_result = [0.0, 0.0, 0.0]
        new_attr = self.cube_node.add_vector_attr("newVectorAttr")
        self.assertCountEqual(new_attr.value, expected_result)

    def test_add_vector_attr_default(self):
        default = [1.0, 2.0, 3.0]
        new_attr = self.cube_node.add_vector_attr(
            "newVectorAttr", default_value=default
        )

        self.assertCountEqual(new_attr.value, default)
        self.assertTrue(new_attr.is_default_value)

    def test_add_vector_attr_initial(self):
        default = [1.0, 2.0, 3.0]
        initial = [4.0, 5.0, 6.0]
        new_attr = self.cube_node.add_vector_attr(
            "newVectorAttr", default_value=default, initial_value=initial
        )
        self.assertCountEqual(new_attr.value, initial)

    def test_add_vector_attr_initial_over_default(self):
        default = [1.0, 2.0, 3.0]
        initial = [4.0, 5.0, 6.0]
        new_attr = self.cube_node.add_vector_attr(
            "newVectorAttr", default_value=default, initial_value=initial
        )

        self.assertCountEqual(new_attr.value, initial)
        self.assertFalse(new_attr.is_default_value)

    def test_add_color_attr(self):
        expected_result = [0.0, 0.0, 0.0]
        new_attr = self.cube_node.add_color_attr("newColorAttr")
        self.assertCountEqual(new_attr.value, expected_result)

    def test_add_color_attr_initial(self):
        initial = [1.0, 2.0, 3.0]
        new_attr = self.cube_node.add_color_attr("newColorAttr", initial_value=initial)
        self.assertCountEqual(new_attr.value, initial)

    def test_add_enum_no_fields(self):
        expected_result = 0
        new_attr = self.cube_node.add_enum_attr("newEnumAttr")
        self.assertEqual(new_attr.value, expected_result)

    def test_add_enum_with_fields(self):
        expected_result = 1
        new_attr = self.cube_node.add_enum_attr(
            "newEnumAttr", fields=["field_one", "field_two"]
        )
        enum_fn_set = OpenMaya.MFnEnumAttribute(new_attr.attribute)
        self.assertEqual(enum_fn_set.getMax(), expected_result)

    def test_add_enum_with_default_value(self):
        expected_result = 1
        new_attr = self.cube_node.add_enum_attr(
            "newEnumAttr", fields=["field_one", "field_two"], default_value=1
        )
        self.assertEqual(new_attr.value, expected_result)
        self.assertTrue(new_attr.is_default_value)

    def test_add_enum_with_initial_value(self):
        expected_result = 1
        new_attr = self.cube_node.add_enum_attr(
            "newEnumAttr", fields=["field_one", "field_two"], initial_value=1
        )
        self.assertEqual(new_attr.value, expected_result)

    def test_add_enum_initial_overrides_default_value(self):
        expected_result = 1
        new_attr = self.cube_node.add_enum_attr(
            "newEnumAttr",
            fields=["field_one", "field_two", "field_three"],
            default_value=2,
            initial_value=1,
        )
        self.assertEqual(new_attr.value, expected_result)
        self.assertFalse(new_attr.is_default_value)


class TestMirageAttrConnectionMethods(unittest.TestCase):
    def setUp(self):
        self.persp_node = mirage.MirageNode.from_name("persp")
        self.cube_node, _ = mirage.MirageNode.from_cmd("polyCube", name="test_cube")

    def tearDown(self):
        try:
            cmds.delete("test_cube*")
        except Exception:
            pass

    def test_connect_cube_translate_from_persp_translate(self):
        # sanity check
        self.assertAlmostEqual(self.cube_node["translateX"].value, 0.0)

        persp_transform = self.persp_node["translateX"]
        self.cube_node["translateX"].connect_from(persp_transform)

        self.assertEqual(self.cube_node["translateX"].value, persp_transform)

        # further sanity check
        self.assertNotEqual(persp_transform.value, self.cube_node["translateX"])

    def test_connect_persp_translate_to_cube_translate(self):
        # sanity check
        self.assertAlmostEqual(self.cube_node["translateX"].value, 0.0)

        self.persp_node["translateX"].connect_to(self.cube_node["translateX"])

        self.assertEqual(
            self.cube_node["translateX"].value, self.persp_node["translateX"]
        )

        # further sanity check
        self.assertNotEqual(
            self.persp_node["translateX"].value, self.cube_node["translateX"]
        )

    def test_break_connections(self):
        # sanity check
        self.assertAlmostEqual(self.cube_node["translateX"].value, 0.0)

        self.persp_node["translateX"].connect_to(self.cube_node["translateX"])

        self.assertEqual(
            self.cube_node["translateX"].value, self.persp_node["translateX"]
        )

        self.cube_node["translateX"].disconnect()

        self.assertAlmostEqual(
            self.cube_node["translateX"].value, self.persp_node["translateX"].value
        )


class TestMirageNodeConvenienceMethods(unittest.TestCase):
    """Tests getting simple (non-compound, non-array) attributes"""

    def setUp(self):
        self.default_sg = mirage.MirageNode.from_name("initialShadingGroup")
        self.cube_node, _ = mirage.MirageNode.from_cmd("polyCube", name="test_cube")

    def tearDown(self):
        for obj_name in (self.cube_node.name, "new_sg*"):
            try:
                cmds.delete(obj_name)
            except Exception:
                pass

    def test_get_shading_group(self):
        self.assertEqual(self.cube_node.shape.shading_group, self.default_sg)

    def test_assign_shading_group(self):
        # equivalent to:
        # `sets -renderable true -noSurfaceShader true -empty -name blinn1SG`
        new_shading_group = mirage.MirageNode.from_cmd(
            "sets", renderable=True, noSurfaceShader=True, empty=True, name="new_sg"
        )[0]

        # sanity check...
        self.assertEqual(self.cube_node.shape.shading_group, self.default_sg)

        self.cube_node.shape.shading_group = new_shading_group
        self.assertNotEqual(self.cube_node.shape.shading_group, self.default_sg)
        self.assertEqual(self.cube_node.shape.shading_group, new_shading_group)


class TestMirageNodeSetLikeInterface(unittest.TestCase):
    def setUp(self):
        self.set_obj = mirage.MirageNode.from_cmd("sets")[0]
        self.cube_one, _ = mirage.MirageNode.from_cmd("polyCube", name="test_cube")
        self.cube_two, _ = mirage.MirageNode.from_cmd("polyCube", name="test_cube")

    def tearDown(self):
        for obj in (self.cube_one, self.cube_two, self.set_obj):
            try:
                cmds.delete(obj.name)
            except Exception:
                pass

    def test_empty_membership(self):
        self.assertFalse(bool(self.set_obj.set_membership))

    def test_assign_membership(self):
        membership_to_assign = (self.cube_one, self.cube_two)
        self.set_obj.set_membership = membership_to_assign
        self.assertCountEqual(self.set_obj.set_membership, membership_to_assign)

    def test_add_member(self):
        self.set_obj.add_set_member(self.cube_one)
        self.assertIn(self.cube_one, self.set_obj.set_membership)

    def test_add_members(self):
        membership_to_add = (self.cube_one, self.cube_two)
        self.set_obj.add_set_members(membership_to_add)
        self.assertCountEqual(self.set_obj.set_membership, membership_to_add)

    def test_remove_member(self):
        membership_to_assign = (self.cube_one, self.cube_two)
        self.set_obj.set_membership = membership_to_assign
        self.set_obj.remove_set_member(self.cube_one)
        self.assertIn(self.cube_two, self.set_obj.set_membership)
        self.assertNotIn(self.cube_one, self.set_obj.set_membership)

    def test_remove_members(self):
        membership_to_assign = (self.cube_one, self.cube_two)
        self.set_obj.set_membership = membership_to_assign
        self.set_obj.remove_set_members((self.cube_one,))
        self.assertIn(self.cube_two, self.set_obj.set_membership)
        self.assertNotIn(self.cube_one, self.set_obj.set_membership)

    def test_clear_membership(self):
        self.set_obj.set_membership = (self.cube_one,)
        self.set_obj.clear_set()
        self.assertFalse(bool(self.set_obj.set_membership))


if __name__ == "__main__":
    import maya.standalone

    maya.standalone.initialize(name="python")
    cmds.scriptEditorInfo(suppressResults=True, suppressWarnings=True)
    unittest.main()
