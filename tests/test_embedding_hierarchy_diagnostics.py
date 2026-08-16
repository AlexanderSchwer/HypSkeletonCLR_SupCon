import os
import tempfile
import unittest

import numpy as np

from tools.hyperbolic_embedding_plot import (
    compute_class_hierarchy,
    format_class_hierarchies,
    render_embedding_diagnostics,
)


class EmbeddingHierarchyDiagnosticsTest(unittest.TestCase):
    def _sample_embeddings(self):
        rng = np.random.RandomState(0)
        centers = np.asarray(
            [
                [0.10, 0.00, 0.00],
                [0.13, 0.02, 0.00],
                [-0.10, 0.00, 0.00],
                [-0.13, -0.02, 0.00],
            ],
            dtype=np.float32,
        )
        embeddings = []
        labels = []
        for class_id, center in enumerate(centers):
            embeddings.append(center + 0.005 * rng.randn(6, 3).astype(np.float32))
            labels.extend([class_id] * 6)
        return np.concatenate(embeddings, axis=0), np.asarray(labels, dtype=np.int64)

    def test_compute_ward_tangent_hierarchy_uses_one_leaf_per_class(self):
        embeddings, labels = self._sample_embeddings()

        hierarchy = compute_class_hierarchy(
            embeddings,
            labels,
            curvature=1.0,
            selected_labels=[0, 1, 2, 3],
            linkage_method="ward_tangent",
        )

        self.assertEqual(hierarchy["class_ids"].tolist(), [0, 1, 2, 3])
        self.assertEqual(hierarchy["class_counts"].tolist(), [6, 6, 6, 6])
        self.assertEqual(hierarchy["linkage_matrix"].shape, (3, 4))
        self.assertEqual(hierarchy["linkage_method"], "ward_tangent")

    def test_compute_hyperbolic_class_hierarchy_uses_poincare_distances(self):
        embeddings, labels = self._sample_embeddings()

        hierarchy = compute_class_hierarchy(
            embeddings,
            labels,
            curvature=1.0,
            selected_labels=[0, 1, 2, 3],
            linkage_method="average_hyperbolic",
        )

        self.assertEqual(hierarchy["linkage_method"], "average_hyperbolic")
        self.assertEqual(hierarchy["linkage_matrix"].shape, (3, 4))
        self.assertEqual(hierarchy["distance_matrix"].shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(hierarchy["distance_matrix"]), 0.0))
        self.assertGreater(hierarchy["distance_matrix"][0, 2], hierarchy["distance_matrix"][0, 1])

    def test_format_class_hierarchy_includes_semantic_groups(self):
        embeddings, labels = self._sample_embeddings()

        table = format_class_hierarchies(
            embeddings,
            labels,
            curvature=1.0,
            selected_labels=[0, 1, 2, 3],
            class_groups={"left": [0, 1], "right": [2, 3]},
            class_names={0: "left a", 1: "left b", 2: "right a", 3: "right b"},
            linkage_methods=["ward_tangent"],
            max_merges=0,
        )

        self.assertIn("Ward Tangent class hierarchy diagnostics", table)
        self.assertIn("left", table)
        self.assertIn("right", table)
        self.assertIn("left a", table)

    def test_format_class_hierarchies_includes_multiple_linkages(self):
        embeddings, labels = self._sample_embeddings()

        table = format_class_hierarchies(
            embeddings,
            labels,
            curvature=1.0,
            selected_labels=[0, 1, 2, 3],
            class_groups={"left": [0, 1], "right": [2, 3]},
            linkage_methods=["ward_tangent", "average_hyperbolic"],
            max_merges=1,
        )

        self.assertIn("Ward Tangent class hierarchy diagnostics", table)
        self.assertIn("Average Hyperbolic class hierarchy diagnostics", table)

    def test_render_embedding_diagnostics_creates_hierarchy_images(self):
        embeddings, labels = self._sample_embeddings()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = render_embedding_diagnostics(
                embeddings,
                labels,
                output_dir=tmpdir,
                epoch=1,
                curvature=1.0,
                projection_methods=["pca"],
                selected_labels=[0, 1, 2, 3],
                color_by=["class_group"],
                class_groups={"left": [0, 1], "right": [2, 3]},
                class_names={0: "left a", 1: "left b", 2: "right a", 3: "right b"},
                render_class_hierarchy=True,
                hierarchy_linkages=["ward_tangent", "average_hyperbolic"],
            )

            ward_paths = [path for path in paths if path.endswith("_ward_tangent_class_hierarchy.png")]
            average_paths = [path for path in paths if path.endswith("_average_hyperbolic_class_hierarchy.png")]
            self.assertEqual(len(ward_paths), 1)
            self.assertEqual(len(average_paths), 1)
            self.assertTrue(os.path.exists(ward_paths[0]))
            self.assertTrue(os.path.exists(average_paths[0]))


if __name__ == "__main__":
    unittest.main()
