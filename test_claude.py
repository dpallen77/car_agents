from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads .env and makes the key available
client = anthropic.Anthropic()  # automatically uses ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(message.content[0].text)