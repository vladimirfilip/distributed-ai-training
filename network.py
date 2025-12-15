import pickle, struct, socket

from message import Message

def send_msg(sock: socket.socket, msg : Message) -> None:
    """
    Serialises the given `msg` and prepends the number of bytes
    :param sock: socket connection
    :param msg: `Message` to serialise
    :return: None
    """
    data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)

def recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Reads `n` bytes from the socket connection
    :param sock: socket connection
    :param n: number of bytes to read
    :return: the bytes read
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Message:
    """
    Read a `Message` from `sock`
    :param sock: socket connection
    :return: the message sent
    """
    (n,) = struct.unpack("!I", recv_exact(sock, 4))
    data = recv_exact(sock, n)
    ret = pickle.loads(data)
    return ret