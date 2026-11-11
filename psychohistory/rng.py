"""Named random channels for the simulator (11 November 2026).

Every place the simulator draws random numbers has its own stream, keyed by (run seed, channel name, simulated year).
A draw that is skipped, added or resized in one place (a mechanism switched off, a scenario, a new feature) therefore
cannot shift the numbers drawn anywhere else, in that year or later: two runs with the same seed differ only through
what the changed mechanism does. Before this the main stream was shared, and pairing was kept by hand (taking draws
whether or not they were used), which the code audit of 10 November 2026 found broken in four places.

    gen = Channels(seed, device)
    gen.at(k)                                   # at the start of simulated year k
    torch.rand(S, C, device=device, generator=gen("coup"))

A channel's generator object is stable (it can be handed to a helper once) and is re-keyed in place by at()."""
from __future__ import annotations

import hashlib

import torch


def channel_seed(seed: int, name: str, k: int) -> int:
    h = hashlib.blake2b(f"{int(seed)}|{name}|{int(k)}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") >> 1          # 63 bits: a valid torch seed on every device


class Channels:
    def __init__(self, seed: int, device):
        self.seed, self.device, self.k, self._g = int(seed), device, 0, {}

    def __call__(self, name: str) -> torch.Generator:
        g = self._g.get(name)
        if g is None:
            g = self._g[name] = torch.Generator(device=self.device).manual_seed(channel_seed(self.seed, name, self.k))
        return g

    def at(self, k: int) -> None:
        self.k = int(k)
        for name, g in self._g.items():
            g.manual_seed(channel_seed(self.seed, name, self.k))

    def initial_seed(self) -> int:               # (the world layer derives its own generators from the run seed)
        return self.seed
