import pickle, struct, socket

from message import Message

def send_msg(sock: socket.socket, msg : Message) -> None:
    data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Message:
    (n,) = struct.unpack("!I", recv_exact(sock, 4))
    data = recv_exact(sock, n)
    ret = pickle.loads(data)
    return ret