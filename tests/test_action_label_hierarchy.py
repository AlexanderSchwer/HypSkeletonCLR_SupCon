import unittest

from tools.action_label_hierarchy import (
    NTU60_CLASS_NAMES,
    hierarchy_class_names,
    hierarchy_leaf_groups,
    hierarchy_leaf_ids,
    hierarchy_leaf_paths,
    list_hierarchies,
)


class ActionLabelHierarchyTest(unittest.TestCase):
    def test_hypskeletonclr_tree_covers_all_zero_based_classes(self):
        class_ids = hierarchy_leaf_ids("hypskeletonclr_ward")

        self.assertEqual(len(class_ids), 60)
        self.assertEqual(sorted(class_ids), list(range(60)))

    def test_hypskeletonclr_tree_exposes_groups_names_and_paths(self):
        class_names = hierarchy_class_names("hypskeletonclr_ward")
        groups = hierarchy_leaf_groups("hypskeletonclr_ward")
        paths = hierarchy_leaf_paths("hypskeletonclr_ward")

        self.assertEqual(class_names[17], "put on glasses")
        self.assertEqual(class_names[54], "hugging")
        self.assertIn(17, groups["personal_and_mundane_tasks"])
        self.assertIn(54, groups["social_and_interactive_actions"])
        self.assertIn(42, groups["full_body_movements"])
        self.assertIn("head_neck_pain", paths[43])
        self.assertIn("head_neck_pain", paths[46])

    def test_class_names_match_rose_lab_ntu60_labels(self):
        self.assertEqual(NTU60_CLASS_NAMES[1], "eat meal")
        self.assertEqual(NTU60_CLASS_NAMES[2], "brush teeth")
        self.assertEqual(NTU60_CLASS_NAMES[13], "put on jacket")
        self.assertEqual(NTU60_CLASS_NAMES[27], "phone call")
        self.assertEqual(NTU60_CLASS_NAMES[39], "cross hands in front")
        self.assertEqual(NTU60_CLASS_NAMES[48], "fan self")
        self.assertEqual(NTU60_CLASS_NAMES[57], "shaking hands")

    def test_lists_only_hypskeletonclr_hierarchy(self):
        self.assertEqual(list_hierarchies(), ["hypskeletonclr_ward"])


if __name__ == "__main__":
    unittest.main()
