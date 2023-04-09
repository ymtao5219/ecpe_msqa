import openai 

class Container:
    def __init__(self, api_key, model='text-davinci-003') -> None:
        self.model = model
        openai.api_key = api_key
    
    def chat(self, text_template, example, temp=0.7, n=2):
        if self.model == 'text-davinci-003':
            prompt = self.prompt_parse(text_template, example)
            print(prompt)
            print(len(prompt.split()))
            completion = openai.Completion.create(model=self.model, prompt=prompt, temperature=temp, n=n, max_tokens=400)
        elif self.model == 'gpt-3.5-turbo':
            message = self.message_parse(text_template, example)
            completion = openai.ChatCompletion.create(model=self.model, messages=message)
        return completion
    
    def prompt_parse(self, template, example):
        assert len(template.fields) == len(example.fields)
        result = template.instruction + '\n'
        for e in template.fields:
            result += ' '.join(e) + '\n'
        result += '--------------------------------\n'
        for i in range(len(example.fields)):
            prefix = template.fields[i][0] # prefix
            result += prefix + ' ' + str(example.fields[i]) + '\n'

        return result
    
    def message_parse(self, template, example):
        pass