import math


class Line:
    def __init__(self, a_x, b_y, c):
        """
        Line in the form of ax+by+c=0
        :param c: offset at the y-axis
        """
        self.a = a_x
        self.b = b_y
        self.c = c

    def distance_to(self, x, y):
        nom = abs(self.a * x + self.b * y + self.c)
        denom = math.sqrt(self.a ** 2 + self.b ** 2)
        return nom / denom


left = Line(4, 1, -1000)
right = Line(-4, 1, 1560)
print(left.distance_to(105, 225))
print(right.distance_to(105, 225))