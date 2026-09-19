# Icosahedron — laws. Esc (or a Close, handled in code) ends the app.
# The on-screen clamp is a computational invariant of Ico.clamp (F32 is
# axiomatic, so it cannot be proven). See SPEC.md.
import Base
import ./main.bend as App

# LAW: an Esc key-down quits, whatever follows it and whatever the state.
law esc_quits:
  for rest: List<Event>
  for s: App.World
  {App.Ico.events(Key{27, True{}} <> rest, s) == None{} : Maybe<&1, App.World>}
