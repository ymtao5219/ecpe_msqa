import openai 
import os
# https://platform.openai.com/account/api-keys

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

document = "c1: Yesterday morning c2: a policeman visited the old man with the lost money, c3: and told him that the thief was caught. c4: The old man was very happy. c5: But he still feels worried, c6: as he doesn’t know how to keep so much money"


text_prompt = 'Given this document:' + document + 'Extract emotion-cause pairs and return the answer as a list of tuples clause numbers of emotion and cause. '

response = openai.Completion.create(model="text-davinci-003", prompt=text_prompt, temperature=0.7, max_tokens=256)

print(response)

