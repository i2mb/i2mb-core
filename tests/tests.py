from i2mb.worlds.square import SquareWorld
from i2mb.engine.agents import AgentList

if __name__ == "__main__":
    pl = AgentList(5)
    for p in pl:
        print(f"Agent id: {p.id}")

    pl = AgentList([1, 2, 5, 10])
    for p in pl:
        print(f"Agent id: {p.id}")
