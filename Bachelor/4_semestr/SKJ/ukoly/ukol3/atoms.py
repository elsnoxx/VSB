
import playground
import random

from typing import List, Tuple, NewType

Pos = NewType('Pos', Tuple[int, int])


class Atom:

    def __init__(self, pos: Pos, vel: Pos, rad: int, col: str):
        """
        Initializer of Atom class

        :param x: x-coordinate
        :param y: y-coordinate
        :param rad: radius
        :param color: color of displayed circle
        """
        try:
            if rad <= 0:
                raise ValueError("Radius must be greater than 0")
            self.pos = pos
            self.vel = vel
            self.rad = rad
            self.col = col
        except Exception as e:
            print(f"Error in Atom.__init__: {e}")

    def to_tuple(self) -> Tuple[int, int, int, str]:
        """
        Returns tuple representing an atom.

        Example: pos = (10, 12,), rad = 15, color = 'green' -> (10, 12, 15, 'green')
        """
        try:
            return (self.pos[0], self.pos[1], self.rad, self.col)
        except Exception as e:
            print(f"Error in Atom.to_tuple: {e}")
            return (0, 0, 0, "error")

    def apply_speed(self, size_x: int, size_y: int):
        """
        Applies velocity `vel` to atom's position `pos`.

        :param size_x: width of the world space
        :param size_y: height of the world space
        """
        try:
            new_x = (self.pos[0] + self.vel[0]) % size_x
            new_y = (self.pos[1] + self.vel[1]) % size_y
            self.pos = (new_x, new_y)
        except Exception as e:
            print(f"Error in Atom.apply_speed: {e}")


class FallDownAtom(Atom):
    """
    Class to represent atoms that are pulled by gravity.
     
    Set gravity factor to ~3.

    Each time an atom hits the 'ground' damp the velocity's y-coordinate by ~0.7.
    
    """
    graviti_factor = 3
    velocity_damp = 0.7

    def apply_speed(self, size_x: int, size_y: int):
        try:
            new_x = (self.pos[0] + self.vel[0]) % size_x
            new_y = self.pos[1] + self.vel[1] + self.graviti_factor

            if new_y >= size_y:
                new_y = size_y
                self.vel = (self.vel[0], -self.vel[1] * self.velocity_damp)

            self.pos = (new_x, new_y)
        except Exception as e:
            print(f"Error in ExampleWorld.add_falldown_atom: {e}")

class ExampleWorld:

    def __init__(self, size_x: int, size_y: int, no_atoms: int, no_falldown_atoms: int):
        """
        ExampleWorld initializer.

        :param size_x: width of the world space
        :param size_y: height of the world space
        :param no_atoms: number of 'bouncing' atoms
        :param no_falldown_atoms: number of atoms that respect gravity
        """

        try:
            self.width = size_x
            self.height = size_y
            self.atoms = self.generate_atoms(no_atoms, no_falldown_atoms)
        except Exception as e:
            print(f"Error in ExampleWorld.__init__: {e}")

    def generate_atoms(self, no_atoms: int, no_falldown_atoms) -> List[Atom|FallDownAtom]:
        """
        Generates `no_atoms` Atom instances using `random_atom` method.
        Returns list of such atom instances.

        :param no_atoms: number of Atom instances
        :param no_falldown_atoms: numbed of FallDownAtom instances
        """
        try:
            list_of_atoms = []
            for _ in range(no_atoms):
                list_of_atoms.append(self.random_atom())
            for _ in range(no_falldown_atoms):
                list_of_atoms.append(self.random_falldown_atom())
            return list_of_atoms
        except Exception as e:
            print(f"Error in ExampleWorld.generate_atoms: {e}")
            return []

    def random_atom(self) -> Atom:
        """
        Generates one Atom instance at random position in world, with random velocity, random radius
        and 'green' color.
        """
        try:
            pos = (random.randint(0, self.width), random.randint(0, self.height))
            vel = (random.randint(-5, 5), random.randint(-5, 5))
            rad = random.randint(5, 20)
            return Atom(pos, vel, rad, 'green')
        except Exception as e:
            print(f"Error in ExampleWorld.random_atom: {e}")
            return Atom((0, 0), (0, 0), 0, 'error')

    def random_falldown_atom(self):
        """
        Generates one FalldownAtom instance at random position in world, with random velocity, random radius
        and 'yellow' color.
        """
        try:
            pos = (random.randint(0, self.width), random.randint(0, self.height))
            vel = (random.randint(-5, 5), random.randint(-5, 5))
            rad = random.randint(5, 20)
            return FallDownAtom(pos, vel, rad, 'yellow')
        except Exception as e:
            print(f"Error in ExampleWorld.random_falldown_atom: {e}")
            return FallDownAtom((0, 0), (0, 0), 0, 'error')

    def add_atom(self, pos_x, pos_y):
        """
        Adds a new Atom instance to the list of atoms. The atom is placed at the point of left mouse click.
        Velocity and radius is random.

        :param pos_x: x-coordinate of a new Atom
        :param pos_y: y-coordinate of a new Atom

        Method is called by playground on left mouse click.
        """
        try:
            vel = (random.randint(-5, 5), random.randint(-5, 5))
            rad = random.randint(5, 20)
            self.atoms.append(Atom((pos_x, pos_y), vel, rad, 'green'))
        except Exception as e:
            print(f"Error in ExampleWorld.add_atom: {e}")

    def add_falldown_atom(self, pos_x, pos_y):
        """
        Adds a new FallDownAtom instance to the list of atoms. The atom is placed at the point of right mouse click.
        Velocity and radius is random.

        Method is called by playground on right mouse click.

        :param pos_x: x-coordinate of a new FallDownAtom
        :param pos_y: y-coordinate of a new FallDownAtom
        """
        try:
            vel = (random.randint(-5, 5), random.randint(-5, 5))
            rad = random.randint(5, 20)
            self.atoms.append(FallDownAtom((pos_x, pos_y), vel, rad, 'yellow'))
        except Exception as e:
            print(f"Error in ExampleWorld.add_falldown_atom: {e}")

    def tick(self):
        """
        Method is called by playground. Sends a tuple of atoms to rendering engine.

        :return: tuple or generator of atom objects, each containing (x, y, radius, color) attributes of atom 
        """
        for atom in self.atoms:
            atom.apply_speed(self.width, self.height)
        return tuple(atom.to_tuple() for atom in self.atoms)


if __name__ == '__main__':
    size_x, size_y = 700, 400
    no_atoms = 2
    no_falldown_atoms = 3

    world = ExampleWorld(size_x, size_y, no_atoms, no_falldown_atoms)

    playground.run((size_x, size_y), world)
