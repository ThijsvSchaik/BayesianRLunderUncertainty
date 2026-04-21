import unittest

from bayesian_rl_under_uncertainty import GridGaussianEnvironment


class TestGridGaussianEnvironment(unittest.TestCase):
    def test_initialization_shapes_and_start(self):
        env = GridGaussianEnvironment(seed=7)

        self.assertEqual(len(env.reward_map), 100)
        self.assertEqual(len(env.reward_map[0]), 100)
        self.assertEqual(len(env.unsafe_mask), 100)
        self.assertEqual(len(env.falling_prob_map), 100)
        self.assertEqual(env.current_state, (0, 0))
        self.assertFalse(env.unsafe_mask[0][0])

    def test_transition_distribution_sums_to_one(self):
        env = GridGaussianEnvironment(seed=1)
        dist = env.transition_distribution((0, 0), "Left")

        self.assertAlmostEqual(sum(dist.values()), 1.0)
        self.assertIn((0, 0), dist)

    def test_step_deterministic_transition(self):
        model = {
            "Left": {"Left": 1.0},
            "Right": {"Right": 1.0},
            "Up": {"Up": 1.0},
            "Down": {"Down": 1.0},
        }
        safe = [[False for _ in range(100)] for _ in range(100)]
        env = GridGaussianEnvironment(seed=2, transition_model=model, unsafe_mask=safe)

        state, _, done, info = env.step("Right")
        self.assertEqual(state, (0, 1))
        self.assertFalse(done)
        self.assertFalse(info["fell"])

    def test_falling_only_on_unsafe_states(self):
        model = {
            "Left": {"Left": 1.0},
            "Right": {"Right": 1.0},
            "Up": {"Up": 1.0},
            "Down": {"Down": 1.0},
        }
        unsafe = [[False for _ in range(100)] for _ in range(100)]
        unsafe[0][1] = True

        falling_prob = [[0.0 for _ in range(100)] for _ in range(100)]
        falling_prob[0][1] = 1.0

        env = GridGaussianEnvironment(
            seed=3,
            transition_model=model,
            unsafe_mask=unsafe,
            falling_prob_map=falling_prob,
        )

        state, _, done, info = env.step("Right")
        self.assertEqual(state, (0, 1))
        self.assertTrue(done)
        self.assertTrue(info["fell"])


if __name__ == "__main__":
    unittest.main()
