from masskrug.worlds.square import SquareWorld
from masskrug.engine.particle import ParticleList

if __name__ == "__main__":
    pl = ParticleList(5)
    for p in pl:
        print(f"Particle id: {p.id}")

    pl = ParticleList([1, 2, 5, 10])
    for p in pl:
        print(f"Particle id: {p.id}")
