import json
import tempfile
import unittest
from pathlib import Path

from terminal_bench.harbor_job import (
    build_job_config,
    compare_summaries,
    summarize_job,
)
from terminal_bench.quant import bits_per_dimension, missing_codebooks, preset_to_pq
from terminal_bench.run_eval import parse_args


class QuantTests(unittest.TestCase):
    def test_d4b8_is_two_bit(self):
        m_sub, nbits = preset_to_pq("d4b8")
        self.assertEqual((m_sub, nbits), (32, 8))
        self.assertEqual(bits_per_dimension(m_sub, nbits), 2.0)

    def test_other_presets(self):
        self.assertEqual(bits_per_dimension(*preset_to_pq("d8b8")), 1.0)
        self.assertEqual(bits_per_dimension(*preset_to_pq("d8b12")), 1.5)
        self.assertEqual(bits_per_dimension(*preset_to_pq("d4b12")), 3.0)
        self.assertEqual(bits_per_dimension(*preset_to_pq("d2b8")), 4.0)

    def test_missing_codebooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "key_cent_layer0_32_8.pq.pt").write_bytes(b"")
            missing = missing_codebooks(root, 32, 8, n_layers=1)
            self.assertEqual(len(missing), 1)
            self.assertTrue(missing[0].name.startswith("val_"))


class HarborConfigTests(unittest.TestCase):
    def test_tblite_job_is_serial_and_points_at_the_local_server(self):
        config = build_job_config(
            job_name="tblite-d4b8",
            jobs_dir="terminal_bench/jobs",
            dataset="tblite",
            base_url="http://host.docker.internal:8000/v1/",
            n_tasks=2,
            task_names=["hello-*"],
        )
        self.assertEqual(config["datasets"][0]["name"], "openthoughts-tblite")
        self.assertEqual(config["datasets"][0]["version"], "2.0")
        self.assertEqual(config["datasets"][0]["n_tasks"], 2)
        self.assertEqual(config["n_concurrent_trials"], 1)
        agent = config["agents"][0]
        self.assertEqual(agent["name"], "terminus-2")
        self.assertEqual(agent["n_concurrent"], 1)
        self.assertEqual(agent["base_url"], "http://host.docker.internal:8000/v1")
        self.assertEqual(agent["model_api_format"], "openai_chat_completions")
        self.assertEqual(agent["kwargs"]["temperature"], 0.0)
        self.assertFalse(agent["kwargs"]["enable_summarize"])
        self.assertEqual(agent["kwargs"]["proactive_summarization_threshold"], 0)
        self.assertIn("host.docker.internal", agent["extra_allowed_hosts"])

    def test_full_tb2_alias(self):
        config = build_job_config(
            job_name="tb2",
            jobs_dir="jobs",
            dataset="terminal-bench@2.0",
            base_url="http://host.docker.internal:8000/v1",
        )
        self.assertEqual(config["datasets"][0]["name"], "terminal-bench")
        self.assertNotIn("n_tasks", config["datasets"][0])


class SummaryTests(unittest.TestCase):
    def test_resolution_rate_and_compare(self):
        def payload(solved, failed):
            return {
                "n_total_trials": solved + failed,
                "stats": {
                    "n_completed_trials": solved + failed,
                    "n_errored_trials": 0,
                    "evals": {
                        "terminus-2__openai/vecinfer__openthoughts-tblite": {
                            "n_trials": solved + failed,
                            "n_errors": 0,
                            "reward_stats": {
                                "reward": {
                                    "1.0": [f"task{i}__aaaaaaa" for i in range(solved)],
                                    "0.0": [f"miss{i}__bbbbbbb" for i in range(failed)],
                                }
                            },
                        }
                    },
                },
                "trial_results": [
                    {
                        "trial_name": "csv-to-parquet__aaaaaaa",
                        "reward_valid": True,
                        "verifier_result": {"rewards": {"reward": 1}},
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            left_path = Path(tmp) / "a" / "result.json"
            right_path = Path(tmp) / "b" / "result.json"
            left_path.parent.mkdir()
            right_path.parent.mkdir()
            left_path.write_text(json.dumps(payload(1, 1)))
            right_path.write_text(json.dumps(payload(2, 0)))
            left = summarize_job(left_path.parent)
            right = summarize_job(right_path)
            self.assertAlmostEqual(left["resolution_rate"], 0.5)
            self.assertAlmostEqual(right["resolution_rate"], 1.0)
            self.assertEqual(left["trials"][0]["task"], "csv-to-parquet")
            text = compare_summaries(left, right)
            self.assertIn("50.00%", text)
            self.assertIn("100.00%", text)


class CliTests(unittest.TestCase):
    def test_defaults_to_lite_and_paper_config(self):
        args = parse_args([])
        self.assertEqual(args.dataset, "tblite")
        self.assertEqual(args.condition, "d4b8")
        self.assertFalse(args.launch)


if __name__ == "__main__":
    unittest.main()
