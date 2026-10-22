"""Real collectives over TCP sockets (full mesh on loopback). Algorithms: ring, star, recursive doubling."""
import socket, struct, threading, time
import numpy as np


class Group:
    def __init__(self, rank, world, base_port, host="127.0.0.1"):
        self.rank, self.world = rank, world
        self.peers = {}
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, base_port + rank)); srv.listen(world)
        # connect to higher ranks, accept from lower ranks
        for j in range(rank + 1, world):
            while True:
                try:
                    s = socket.create_connection((host, base_port + j)); break
                except OSError:
                    time.sleep(0.02)
            s.sendall(struct.pack("!i", rank)); self.peers[j] = s
        for _ in range(rank):
            c, _ = srv.accept(); (j,) = struct.unpack("!i", c.recv(4)); self.peers[j] = c
        srv.close()
        for s in self.peers.values():
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20); s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.bytes_sent = 0

    def _send(self, j, arr):
        self.bytes_sent += arr.nbytes; self.peers[j].sendall(memoryview(arr).cast("B"))

    def _recv(self, j, out):
        mv = memoryview(out).cast("B"); n = 0
        while n < len(mv):
            k = self.peers[j].recv_into(mv[n:]); 
            if k == 0: raise ConnectionError("peer closed")
            n += k

    def sendrecv(self, to, arr, frm, out):
        t = threading.Thread(target=self._send, args=(to, arr)); t.start()
        self._recv(frm, out); t.join()

    def barrier(self):
        x = np.zeros(1, np.float32); self.allreduce(x, "star")

    def allreduce(self, x, algo="ring"):
        """In-place sum all-reduce on a 1-D float32 array."""
        return {"ring": self._ring, "star": self._star, "rd": self._rd}[algo](x)

    def _star(self, x):
        n, r = self.world, self.rank
        if n == 1: return x
        if r == 0:
            tmp = np.empty_like(x)
            for j in range(1, n): self._recv(j, tmp); x += tmp
            for j in range(1, n): self._send(j, x)
        else:
            self._send(0, x); self._recv(0, x)
        return x

    def _rd(self, x):  # recursive doubling, power-of-two world
        n, r = self.world, self.rank
        assert n & (n - 1) == 0, "rd needs power-of-two world"
        tmp = np.empty_like(x); d = 1
        while d < n:
            p = r ^ d; self.sendrecv(p, x, p, tmp); x += tmp; d <<= 1
        return x

    def _ring(self, x):
        n, r = self.world, self.rank
        if n == 1: return x
        bounds = np.linspace(0, x.size, n + 1).astype(int)
        ch = lambda i: x[bounds[i % n]:bounds[i % n + 1]]
        right, left = (r + 1) % n, (r - 1) % n
        tmp = np.empty(int(np.diff(bounds).max()), np.float32)
        for s in range(n - 1):  # reduce-scatter
            si, ri = r - s, r - s - 1
            rc = ch(ri); buf = tmp[:rc.size]
            self.sendrecv(right, np.ascontiguousarray(ch(si)), left, buf); rc += buf
        for s in range(n - 1):  # all-gather
            si, ri = r + 1 - s, r - s
            rc = ch(ri); buf = tmp[:rc.size]
            self.sendrecv(right, np.ascontiguousarray(ch(si)), left, buf); rc[:] = buf
        return x

    def close(self):
        for s in self.peers.values(): s.close()
