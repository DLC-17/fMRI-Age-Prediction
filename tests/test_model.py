import unittest
import os
import json
import numpy as np

class TestModelBooster(unittest.TestCase):
    def test_model_file_exists(self):
        model_path = "xgboost_fmri.json"
        self.assertTrue(os.path.exists(model_path), "xgboost_fmri.json should exist")
        
        with open(model_path) as f:
            data = json.load(f)
            
        trees = data["learner"]["gradient_booster"]["model"]["trees"]
        self.assertEqual(len(trees), 100, "Model should contain exactly 100 boosted trees")
        
        base_score = float(data["learner"]["learner_model_param"]["base_score"].strip("[]"))
        self.assertAlmostEqual(base_score, 11.30488, places=4, msg="Base score should match dataset mean ~11.3y")

    def test_web_assets_exist(self):
        required_assets = [
            "index.html",
            "assets/xgb_model.js",
            "assets/archetypes.js",
            "assets/fig7_predicted_vs_actual_scatter.png",
            "assets/fig8_bias_correction_histograms.png",
            "assets/fig2_age_distribution_raw.png",
            "assets/fig3_resampled_age_distribution.png",
            "assets/fig4_ridge_alpha_tuning.png",
            "assets/synthetic_data_attempt.png"
        ]
        for asset in required_assets:
            self.assertTrue(os.path.exists(asset), f"Missing required web asset: {asset}")

    def test_pure_tree_evaluation(self):
        """Validates that manual tree traversal yields sane pediatric age bounds [5, 22]."""
        with open("xgboost_fmri.json") as f:
            data = json.load(f)

        trees = data["learner"]["gradient_booster"]["model"]["trees"]
        base_score = float(data["learner"]["learner_model_param"]["base_score"].strip("[]"))

        def predict_pure(features):
            score = base_score
            for t in trees:
                left = t["left_children"]
                right = t["right_children"]
                cond = t["split_conditions"]
                feat = t["split_indices"]
                weights = t["base_weights"]
                node = 0
                while left[node] != -1:
                    f_idx = feat[node]
                    val = features[f_idx]
                    if val < cond[node]:
                        node = left[node]
                    else:
                        node = right[node]
                score += weights[node]
            return score

        # Test zero vector
        pred_zero = predict_pure([0.0] * 40)
        self.assertGreater(pred_zero, 5.0)
        self.assertLess(pred_zero, 21.9)

        # Test bounded range
        rng = np.random.RandomState(42)
        for _ in range(20):
            vec = rng.randn(40).tolist()
            pred = predict_pure(vec)
            self.assertGreater(pred, 4.0)
            self.assertLess(pred, 23.0)

if __name__ == "__main__":
    unittest.main()
