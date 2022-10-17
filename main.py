import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock


kivy.require("1.9.0")#2.0 moze nie dzialac na wszystkich smartfonach


class MyRoot(BoxLayout): #boxlayout ktory w srodku ma gridlayout

    from bot import chat
    
    
    def __init__(self): #UI with functionalities
        super(MyRoot, self).__init__()

    def send_message(self):
        message = self.message_text.text
        self.chat_text.text += "You: " + message + "\n"
        #self.canvas.ask_update()
        self.message_text.text = ""
        tekst = self.chat(message)
        self.chat_text.text += tekst

    
class Interface(App):

    def build(self):
        return MyRoot()


interface = Interface()
interface.run()
    
                 
