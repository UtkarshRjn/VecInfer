import json
import unittest

from terminal_bench.terminus_format import is_terminus_chat, repair_terminus_response


class TerminusFormatTests(unittest.TestCase):
    def test_markdown_python_becomes_a_command(self):
        draft = (
            "Here is a script:\n\n```python\n"
            "from pathlib import Path\n"
            "Path('/app/summary.csv').write_text('severity,count\\n')\n"
            "```\n"
        )
        repaired = json.loads(repair_terminus_response(draft))
        keystrokes = repaired["commands"][0]["keystrokes"]
        self.assertTrue(keystrokes.startswith("python - << 'PY'\n"))
        self.assertIn("summary.csv", keystrokes)
        self.assertTrue(keystrokes.endswith("PY\n"))

    def test_shell_prompt_is_stripped(self):
        draft = "```bash\nroot@box:/app# python log_analyzer.py\n```\n"
        repaired = json.loads(repair_terminus_response(draft))
        self.assertEqual(repaired["commands"][0]["keystrokes"], "python log_analyzer.py\n")

    def test_bare_tee_does_not_swallow_the_file_write(self):
        payload = {
            "analysis": "counts are ready",
            "plan": "write the csv",
            "commands": [
                {"keystrokes": "tee /app/summary.csv\n", "duration": 0.1},
                {
                    "keystrokes": 'tee /app/summary.csv <<< "severity,count\\nERROR,4\\nWARNING,3\\nINFO,8"\n',
                    "duration": 1.0,
                },
            ],
            "task_complete": True,
        }
        repaired = json.loads(repair_terminus_response(json.dumps(payload)))
        self.assertEqual(len(repaired["commands"]), 1)
        written = repaired["commands"][0]["keystrokes"]
        self.assertIn("severity,count\nERROR,4\nWARNING,3\nINFO,8", written)
        self.assertNotIn("\\n", written)

    def test_cd_dotdot_is_a_real_command(self):
        payload = {
            "analysis": "leave logs",
            "plan": "write the csv one directory up",
            "commands": [
                {"keystrokes": "cd..\n", "duration": 0.1},
                {"keystrokes": 'echo "ERROR,4" >> summary.csv\n', "duration": 0.1},
            ],
        }
        repaired = json.loads(repair_terminus_response(json.dumps(payload)))
        self.assertEqual(repaired["commands"][0]["keystrokes"], "cd ..\n")
        self.assertIn("summary.csv", repaired["commands"][1]["keystrokes"])

    def test_native_json_is_kept(self):
        payload = {
            "analysis": "done",
            "plan": "stop",
            "commands": [{"keystrokes": "ls\n", "duration": 1}],
        }
        repaired = json.loads(repair_terminus_response("Sure.\n" + json.dumps(payload)))
        self.assertEqual(repaired["commands"][0]["keystrokes"], "ls\n")

    def test_invalid_escape_does_not_drop_the_commands(self):
        raw = (
            '{"analysis": "interrupted by `\\echo $PYTHONPATH`", '
            '"plan": "uninstall the wrong package", '
            '"commands": [{"keystrokes": "pip uninstall app\\n", "duration": 1.0}], '
            '"task_complete": false}'
        )
        repaired = json.loads(repair_terminus_response(raw))
        self.assertEqual(repaired["commands"][0]["keystrokes"], "pip uninstall app\n")
        self.assertIn("echo", repaired["analysis"])

    def test_tail_follow_and_a_foreground_server_do_not_swallow_later_keys(self):
        payload = {
            "analysis": "start the worker",
            "plan": "launch it, then read the log",
            "commands": [
                {"keystrokes": "python3 worker_queue.py\n", "duration": 1.0},
                {"keystrokes": "tail -f /app/logs/worker.log\n", "duration": 0.5},
                {"keystrokes": "python3 worker_queue.py status\n", "duration": 0.5},
            ],
        }
        repaired = json.loads(repair_terminus_response(json.dumps(payload)))
        self.assertEqual(repaired["commands"][0]["keystrokes"], "python3 worker_queue.py &\n")
        self.assertEqual(repaired["commands"][1]["keystrokes"], "tail -n 80 /app/logs/worker.log\n")
        self.assertEqual(repaired["commands"][2]["keystrokes"], "python3 worker_queue.py status\n")

    def test_truncated_json_keeps_the_finished_command(self):
        raw = (
            '{\n  "analysis": "Need the postal code.",\n'
            '  "plan": "Print the last field.",\n'
            '  "commands": [\n'
            '    {"keystrokes": "awk \'{print $NF}\' data.csv > temp.csv\\n", "duration": 1.0},\n'
            '    {"keystrokes": "sed'
        )
        repaired = json.loads(repair_terminus_response(raw))
        self.assertEqual(len(repaired["commands"]), 1)
        self.assertIn("temp.csv", repaired["commands"][0]["keystrokes"])
        self.assertFalse(repaired["task_complete"])

    def test_summary_chat_is_not_terminus(self):
        self.assertFalse(
            is_terminus_chat([{"role": "user", "content": "Answer each question in detail."}])
        )
        self.assertTrue(
            is_terminus_chat(
                [{"role": "user", "content": 'Reply as JSON with "keystrokes" fields.'}]
            )
        )


if __name__ == "__main__":
    unittest.main()
