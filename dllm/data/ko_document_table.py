"""
Data loader for jp1924/KoDocumentTableVisualSFT dataset
"""
import json
from typing import Optional
from datasets import load_dataset, DatasetDict


def _parse_content(content: str) -> str:
    """Parse the content field which is a JSON string containing text and image types.

    Args:
        content: JSON string like '[{"type": "image"}, {"type": "text", "text": "..."}]'

    Returns:
        Extracted text content
    """
    try:
        parsed = json.loads(content)
        # Extract text from all text-type entries
        texts = [item["text"] for item in parsed if item.get("type") == "text"]
        return " ".join(texts).strip()
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, return content as-is
        return content.strip()


def load_dataset_ko_document_table(
    dataset_name_or_path: str = "jp1924/KoDocumentTableVisualSFT",
    test_size: float = 0.1,
) -> DatasetDict:
    """Load the jp1924/KoDocumentTableVisualSFT dataset for vision-language training.

    This dataset contains Korean document/table images with QA conversations.

    Dataset structure:
        - id: int
        - image: PIL.Image
        - conversations: list of dicts with 'role' and 'content' fields
            - content is a JSON string: [{"type": "image"}, {"type": "text", "text": "..."}]
        - metadata: dict (not used)

    Returns a DatasetDict with:
        - messages: list of dicts (OpenAI format) with parsed text content
        - image: PIL.Image (original image)

    Args:
        dataset_name_or_path: HuggingFace dataset path
        test_size: Proportion for test split (default: 0.1)
    """
    dataset = load_dataset(dataset_name_or_path)

    def map_fn(example):
        """Convert conversations to OpenAI-compatible format and keep image."""
        conversations = example.get("conversations", [])

        # Parse each conversation turn
        messages = []
        for turn in conversations:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            # Parse the JSON content to extract text
            parsed_text = _parse_content(content)

            messages.append({
                "role": role,
                "content": parsed_text
            })

        return {
            "messages": messages,
            "image": example["image"],  # Keep PIL Image as-is
        }

    # Apply transformation
    dataset = dataset.map(
        map_fn,
        remove_columns=[col for col in dataset["train"].column_names if col not in ["image"]],
        num_proc=4,
        desc="Processing KoDocumentTableVisualSFT"
    )

    # Create train/test split
    dataset = dataset["train"].train_test_split(test_size=test_size, seed=42)

    return dataset


if __name__ == "__main__":
    # Test the loader
    dataset = load_dataset_ko_document_table()
    print(f"Dataset splits: {list(dataset.keys())}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")

    # Show first example
    example = dataset["train"][0]
    print("\nFirst example:")
    print(f"  Image size: {example['image'].size}")
    print(f"  Messages ({len(example['messages'])} turns):")
    for i, msg in enumerate(example["messages"][:4], 1):  # Show first 4 turns
        print(f"    {i}. [{msg['role']}]: {msg['content'][:100]}...")
