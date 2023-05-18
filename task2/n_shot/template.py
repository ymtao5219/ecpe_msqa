class Template:
    def __init__(self, instruction) -> None:
        self.instruction = instruction
        self.fields = []
    
    def __call__(self, **kw):
        # self.fields.append(q)
        for arg in kw.values():
            self.fields.append(arg)
        return self.fields


class Example:
    def __init__(self, **kw) -> None:
        self.fields = []
        for arg in kw.values():
            self.fields.append(arg)

    def __call__(self):
        return self.fields