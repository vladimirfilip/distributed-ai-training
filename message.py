from enum import Enum

class MessageType(Enum):
    HELLO = 1
    PARAMS = 2
    GRADS = 3
    STOP = 4

class Message:
    def __init__(self, message_type: MessageType, data: object):
        self.message_type = message_type
        self.data = data