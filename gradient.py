from math import isclose

def grad_and_inter( x0, y0, x1, y1 ):
    """ Return gradient and intercept in y = gradient * x + intercept
        for a line through (x0,y0) and (x1, y1) """
    try:
        grad = (y1-y0) / (x1-x0)
        return grad, y0 - grad * x0  # gradient & intercept
    except ZeroDivisionError:
        return float('Inf'), x0      # if vertical Inf and x

def frame_intersect( xlo, ylo, xhi, yhi, grad, inter ):
    """ Return the 2 points on the bounding box (xlo, ylo), (xhi, yhi)
        where a line (y = grad * x + inter) intercepts the box. """

    if grad == 0.0:    # y = inter for all x 
        return [ (xlo, inter), (xhi, inter) ]

    if grad == float('Inf'):  # x = inter for all y
        return [(inter, ylo), (inter, yhi) ]

    def y_func( x ):
        y = grad * x + inter
        if (y < ylo) or (y > yhi):
            return None   # Return None if outside the bounds
        return ( x, y )  # Return x and the calculated y

    def x_func( y ):
        x = (y - inter) / grad
        if (x < xlo) or (x>xhi):
            return None   # Return None if outside the bounds
        return ( x, y )  # Return calculated x and y

    func = [ x_func, x_func, y_func, y_func ] # Iterate through the 4 funcs
    param = [ ylo, yhi, xlo, xhi ]            # with the 4 parameters.

    res = []
    first_x = float('Inf')  # first_x must be a float for isclose below.
    for f, p in zip(func, param):
        coords  = f(p)
        if coords is None:
            continue
        if not isclose(first_x, coords[0]):  
            # If the x in coord is the same as the first x don't append it.
            # If one (or both) coords are a corner they'll be repeated.
            res.append(coords)
            if len(res)>1:
                break  # Once two coords found break.
            first_x = coords[0]

    return res

def constant_frame( xlo, ylo, xhi, yhi ):
    """ Returns a function that uses a constant bounding box. """

    def user_func( x0, y0, x1, y1 ):
        return frame_intersect( xlo, ylo, xhi, yhi, *grad_and_inter( x0, y0, x1, y1 ) )

    return user_func

do_640x480 = constant_frame( 0., 0., 640., 480. )

print( 247., 50., 247., 10., do_640x480( 247., 50., 247., 10. ) ) # Vertical
print( 123., 50., 456., 50. , do_640x480( 123., 50., 456., 50. ) ) # Horizontal
print( 10., 50., 450, 450., do_640x480( 10., 50., 450, 450.) )
print( 10., 5., 630, 300., do_640x480( 10., 5., 630, 300.) )
print( 10., 300., 530, 5., do_640x480( 10., 300., 530, 5.) )
print( 10., 5., 500, 400., do_640x480( 10., 5., 500, 400.) )
print( 10., 10.,500, 300., do_640x480( 10., 10., 500, 300.) )