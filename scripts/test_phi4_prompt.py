from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
conversation = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [
        {"type": "audio", "audio": "fake_path.wav"},
        {"type": "text", "text": "Transcribe the following Nepali audio into Nepali text."}
    ]}
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print("ACTUAL PROMPT IS:")
print(repr(prompt))
