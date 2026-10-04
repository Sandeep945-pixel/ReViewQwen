"""Offline interface tests; no checkpoint download or benchmark evaluation."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import infer


class InferenceTests(unittest.TestCase):
    def test_explicit_labels(self):
        for label in (-1, 0, 1):
            self.assertEqual(infer.parse_label(f"Label:\n{label}\nExplanation: evidence"), label)
            self.assertEqual(infer.parse_label(str(label)), label)

    def test_does_not_guess_from_numbers_or_conflicting_labels(self):
        for response in ("It arrived 1 day late.", "Label: 10", "Label: 0\nLabel: 1", "", "Unknown"):
            self.assertIsNone(infer.parse_label(response))

    def test_relative_paths_and_message_order(self):
        example = infer.load_example(ROOT / "demo/example.json")
        self.assertTrue(Path(example["buyer_image"]).is_absolute())
        messages = infer.build_messages(example, ["buyer-fixture", "seller-fixture"])
        items = messages[1]["content"]
        self.assertEqual(items[1]["image"], "buyer-fixture")
        self.assertEqual(items[3]["image"], "seller-fixture")
        self.assertTrue(items[0]["text"].startswith("Buyer Description:"))

    def test_missing_image_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"seller_description": "test", "buyer_review": "test", "seller_image": "missing.png", "buyer_image": "missing.png"}))
            with self.assertRaisesRegex(ValueError, "Image not found"):
                infer.load_example(path)

    def test_validate_only_never_calls_model(self):
        with patch.object(infer, "run_model") as model, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(infer.main(["--example", str(ROOT / "demo/example.json"), "--validate-only"]), 0)
            model.assert_not_called()

    def test_corrupt_image_fails_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "broken.png"
            image.write_text("not an image")
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"seller_description": "test", "buyer_review": "test", "seller_image": image.name, "buyer_image": image.name}))
            with patch.object(infer, "run_model") as model, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(infer.main(["--example", str(path)]), 1)
                model.assert_not_called()

    def test_pinned_adapter_and_completion_only_decoding(self):
        class Inputs(dict):
            input_ids = [[10, 11]]
            def to(self, device):
                return self

        model = Mock()
        model.get_input_embeddings.return_value = types.SimpleNamespace(weight=types.SimpleNamespace(device="cpu"))
        model.generate.return_value = [[10, 11, 21, 22]]
        processor = Mock()
        processor.apply_chat_template.return_value = "formatted prompt"
        processor.return_value = Inputs(input_ids=[[10, 11]])
        processor.batch_decode.return_value = ["Label: 0\nExplanation: The colours differ."]
        torch = types.ModuleType("torch")
        torch.float32, torch.float16 = "float32", "float16"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.inference_mode = contextlib.nullcontext
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=Mock(return_value=processor))
        transformers.Qwen2VLForConditionalGeneration = types.SimpleNamespace(from_pretrained=Mock(return_value="base"))
        peft = types.ModuleType("peft")
        peft.PeftConfig = types.SimpleNamespace(from_pretrained=Mock(return_value=types.SimpleNamespace(base_model_name_or_path=infer.BASE_MODEL)))
        peft.PeftModel = types.SimpleNamespace(from_pretrained=Mock(return_value=model))
        args = argparse.Namespace(device="cpu", max_new_tokens=100, pixels=424)
        example = infer.load_example(ROOT / "demo/example.json")
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "peft": peft}), contextlib.redirect_stderr(io.StringIO()):
            result = infer.run_model(example, ["buyer", "seller"], args)
        peft.PeftModel.from_pretrained.assert_called_once_with("base", infer.ADAPTER, revision=infer.ADAPTER_REVISION, is_trainable=False)
        processor.batch_decode.assert_called_once_with([[21, 22]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(result["category"], "discrepancy")
        self.assertEqual(result["adapter_revision"], infer.ADAPTER_REVISION)
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])


if __name__ == "__main__":
    unittest.main()
