QUIT = object()

class Scene:
    def __init__(self, fonts):
        self.fonts = fonts
        self.next = None      

    def handle_event(self, e): pass
    def update(self): pass
    def draw(self, screen): pass
    def on_exit(self): pass   