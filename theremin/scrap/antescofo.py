"""
Antescofo OSC demo controller.

This script sends OSC messages to an `antescofo~` object (in Max/MSP or Pure Data)
to dynamically change tempo and trigger chords as defined in a companion
Antescofo score file (see `chords.asco` in the same folder).

Defaults assume Antescofo is running locally and listening on port 9000 with
the following OSC addresses in the score:
  - /tempo_change -> variable $tempo
  - /play_next    -> variable $play

Usage (as a script):
  python -m theremin.scrap.antescofo           # run the demo sequence
  python -m theremin.scrap.antescofo --help    # see options

Note: Ensure `python-osc` is installed in your environment.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable, Sequence

try:
    from pythonosc import osc_message_builder
    from pythonosc import udp_client
except Exception as e:  # pragma: no cover - import guidance
    print(
        "Missing dependency `python-osc`. Install it with: pip install python-osc",
        file=sys.stderr,
    )
    raise


DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 9000
ADDR_TEMPO = "/tempo_change"
ADDR_PLAY = "/play_next"


class OscClient:
    """Thin wrapper around python-osc UDPClient with convenience helpers."""

    def __init__(self, ip: str = DEFAULT_IP, port: int = DEFAULT_PORT):
        self.ip = ip
        self.port = int(port)
        self._client = udp_client.UDPClient(self.ip, self.port)

    def send(self, address: str, value):
        msg = osc_message_builder.OscMessageBuilder(address=address)
        msg.add_arg(value)
        built = msg.build()
        self._client.send(built)
        print(f"Sent OSC -> {self.ip}:{self.port} {address} {value}")


def set_tempo(client: OscClient, bpm: float | int):
    client.send(ADDR_TEMPO, float(bpm))


def play_chord(client: OscClient, chord_id: int):
    # chord_id expected by the Antescofo score's $play variable
    client.send(ADDR_PLAY, int(chord_id))


def demo(
    ip: str = DEFAULT_IP,
    port: int = DEFAULT_PORT,
    tempos: Sequence[int] | None = None,
    chords: Sequence[int] | None = None,
    tempo_pause: float = 3.0,
    chord_pause: float = 2.0,
):
    """Run the demo sequence: sweep tempos then trigger chord IDs.

    - tempos: list of BPM values to send to /tempo_change
    - chords: list of chord IDs to send to /play_next
    """
    if tempos is None:
        tempos = [60, 90, 120]
    if chords is None:
        chords = [1, 2, 3]

    client = OscClient(ip, port)

    print("Sending demo commands to Antescofo...")
    for bpm in tempos:
        print(f"\nSetting tempo to {bpm} BPM...")
        set_tempo(client, bpm)
        time.sleep(tempo_pause)

    for cid in chords:
        print(f"\nTrigger chord {cid}...")
        play_chord(client, cid)
        time.sleep(chord_pause)

    print("\nDemo complete.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send OSC to Antescofo for a small demo.")
    p.add_argument(
        "--ip",
        default=DEFAULT_IP,
        help="IP of the host running Antescofo (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="OSC port (default: 9000)"
    )

    p.add_argument(
        "--tempos",
        type=float,
        nargs="*",
        default=None,
        help="Tempo values (BPM) to send in order. Default: 60 90 120",
    )
    p.add_argument(
        "--chords",
        type=int,
        nargs="*",
        default=None,
        help="Chord IDs to trigger in order. Default: 1 2 3",
    )
    p.add_argument(
        "--tempo-pause", type=float, default=3.0, help="Pause (s) between tempo changes"
    )
    p.add_argument(
        "--chord-pause",
        type=float,
        default=2.0,
        help="Pause (s) between chord triggers",
    )

    p.add_argument(
        "--send-tempo",
        type=float,
        help="Send a single tempo value and exit (skips demo sequence)",
    )
    p.add_argument(
        "--send-chord",
        type=int,
        help="Send a single chord ID and exit (skips demo sequence)",
    )

    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)

    # Single-shot commands
    if args.send_tempo is not None or args.send_chord is not None:
        client = OscClient(args.ip, args.port)
        if args.send_tempo is not None:
            set_tempo(client, args.send_tempo)
        if args.send_chord is not None:
            play_chord(client, args.send_chord)
        return

    # Demo sequence
    demo(
        ip=args.ip,
        port=args.port,
        tempos=args.tempos,
        chords=args.chords,
        tempo_pause=args.tempo_pause,
        chord_pause=args.chord_pause,
    )


if __name__ == "__main__":
    main()
