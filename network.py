import pickle, struct, socket


def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
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


def recv_msg(sock: socket.socket):
    (n,) = struct.unpack("!I", recv_exact(sock, 4))
    data = recv_exact(sock, n)
    ret = pickle.loads(data)
    return ret
