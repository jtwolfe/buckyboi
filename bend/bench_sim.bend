import Base
import ./main.bend as App

def Bench.loop(n: Nat, s: App.World) -> App.World:
  match n:
    case 0n:
      s
    case 1n+p:
      Bench.loop(p, App.Ico.advance(s))

def Bench.report(s: App.World) -> IO(Unit):
  App.World{+cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  IO.print("bend done cx=" ++ F32.show(cx))

def main() -> IO(Unit):
  Bench.report(Bench.loop(U32.to_nat(1000000), App.Ico.init()))
