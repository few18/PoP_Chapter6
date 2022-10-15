"""A module providing numerical solvers for nonlinear equations."""

import numpy as np


class ConvergenceError(Exception):
    """Exception raised if a solver fails to converge."""

    pass


def newton_raphson(f, df, x_0, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using Newton-Raphson iteration.

    Solve f==0 using Newton-Raphson iteration.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the iteration.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using Newton iteration.
    """

    iteration = 0
    while np.abs(f(x_0)) > eps:
        x_0 = x_0 - (f(x_0) / df(x_0))
        iteration += 1
        if iteration > max_its:
            raise ConvergenceError
    return x_0


def bisection(f, x_0, x_1, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using bisection.

    Solve f==0 using bisection starting with the interval [x_0, x_1]. f(x_0)
    and f(x_1) must differ in sign.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    x_0 : float
        The left end of the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using bisection.
    """

    iteration = 0
    x_mid = (x_0 + x_1)/2
    while np.abs(f(x_mid)) > eps:
        x_mid = (x_0 + x_1)/2
        f_xmid = f(x_mid)
        f_x0 = f(x_0)
        f_x1 = f(x_1)
        if (f_x0 > 0) and (f_x1 > 0):
            raise ValueError("f(x) positive for both endpoints.")
        elif (f_x0 < 0) and (f_x1 < 0):
            raise ValueError("f(x) negative for both endpoints.")
        if f_xmid < 0 and f_x0 < 0:
            x_0 = x_mid
        elif f_xmid > 0 and f_x0 > 0:
            x_0 = x_mid
        elif f_xmid > 0 and f_x1 > 0:
            x_1 = x_mid
        elif f_xmid < 0 and f_x1 < 0:
            x_1 = x_mid
        iteration += 1
        if iteration > max_its:
            raise ConvergenceError
    return x_mid


def solve(f, df, x_0, x_1, eps=1.0e-5, max_its_n=20, max_its_b=20):
    """Solve a nonlinear equation.

    solve f(x) == 0 using Newton-Raphson iteration, falling back to bisection
    if the former fails.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the Newton-Raphson iteration, and left end of
        the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its_n : int
        The maximum number of iterations to be taken before the newton-raphson
        solver is taken to have failed.
    max_its_b : int
        The maximum number of iterations to be taken before the bisection
        solver is taken to have failed.

    Returns
    -------
    float
        The approximate root.
    """

    try:
        x = newton_raphson(f, df, x_0, eps, max_its_n)
    except ConvergenceError:
        x = bisection(f, x_0, x_1, eps, max_its_b)
    return x
