# Backpropagating through Eurocode 3 with Blueprints

[Blueprints](https://github.com/Blueprints-org/blueprints) evaluates engineering
formulas in scalar Python. It does not carry an autodiff tape, and we do not ask
it to grow one. Normax calls the real check in the forward pass and supplies the
small reverse rule the optimizer needs. The law speaks Python. The gradient
still gets a reply.

This note describes the exact slice implemented in
[`normax/sizing/blueprint.py`](../normax/sizing/blueprint.py). It is a research
model of cross-section resistance from
[Eurocode 3, Part 1-1: general rules and rules for
buildings](https://eurocodes.jrc.ec.europa.eu/EN-Eurocodes/eurocode-3-design-steel-structures),
not a general Eurocode 3 or certification implementation.

## The tractable slice

The headline studies make four choices before optimization:

- S355 steel and the configured value of $\gamma_{M0}$ are fixed.
- Every member is a circular hollow section (CHS).
- Every CHS is fixed at the Class 3 diameter-to-thickness limit.
- Outer diameter $d$ is the only section variable.

For Eurocode 3 Table 5.2, the Class 3 limit used by Normax is

$$
r = \frac{d}{t} = 90\epsilon^2,
\qquad
\epsilon^2 = \frac{235}{f_y}.
$$

With S355 steel, this ratio is precomputed as

$$
r = 90\frac{235}{355} \approx 59.58.
$$

The optimizer chooses $d$. The wall thickness and inner diameter follow:

$$
t = \frac{d}{r},
\qquad
d_i = d - 2t = d\left(1-\frac{2}{r}\right).
$$

Fixing $r$ makes classification true by construction. It also removes a
discrete class switch from the differentiated problem. That is the bargain:
one smooth Class 3 family in exchange for a tractable and explicit derivative.
Catalog selection, other section families, and changes of class remain outside
the project scope.

## The forward rule

At fixed $r$, CHS area and elastic section modulus are monomials in $d$:

$$
\begin{aligned}
A(d) &= c_A d^2,
& c_A &= \frac{\pi}{r}\left(1-\frac{1}{r}\right), \\
W_{\mathrm{el}}(d) &= c_W d^3,
& c_W &= \frac{\pi}{32}\left[1-\left(1-\frac{2}{r}\right)^4\right].
\end{aligned}
$$

Blueprints supplies the Class 3 axial and elastic bending resistances. Normax
combines them into the cross-section utilization

$$
U(d,N,M)
= \frac{|N|}{A(d)f_y/\gamma_{M0}}
+ \frac{M}{W_{\mathrm{el}}(d)f_y/\gamma_{M0}}
= \frac{a}{d^2}+\frac{b}{d^3},
$$

where

$$
a=\frac{\gamma_{M0}|N|}{c_A f_y},
\qquad
b=\frac{\gamma_{M0}M}{c_W f_y}.
$$

For a CHS, the demand moment is the larger end-moment magnitude,

$$
M=\max_{j\in\{1,2\}}\sqrt{M_{y,j}^2+M_{z,j}^2}.
$$

Because $U$ decreases strictly with positive $d$, the fully worked diameter
$d_\star$ is the unique root of $U(d_\star,N,M)=1$. Normax finds it by
bisection, then applies the minimum catalog diameter:

$$
d_{\mathrm{out}}=\max(d_\star,d_{\min}).
$$

## The backward rule

The useful partial derivatives are short enough to inspect:

$$
\begin{aligned}
U_d &= -\frac{2a}{d^3}-\frac{3b}{d^4}, \\
U_N &= \frac{\gamma_{M0}\,\mathrm{sign}(N)}{c_A f_y d^2}, \\
U_M &= \frac{\gamma_{M0}}{c_W f_y d^3}.
\end{aligned}
$$

At an unclamped sizing root, implicit differentiation gives

$$
\frac{\partial d_\star}{\partial z}
=-\frac{U_z}{U_d},
\qquad z\in\{N,M\}.
$$

The two branches have complementary derivatives:

| Branch | Diameter derivative | Reported-utilization derivative |
|---|---|---|
| $d_\star\ge d_{\min}$ | implicit rule above | zero, because $U(d_\star)=1$ |
| $d_\star<d_{\min}$ | zero | explicit partials at $d_{\min}$ |

A held-diameter check is simpler. It uses the explicit partials directly, so a
cotangent $\bar U$ returns $\bar d=\bar U U_d$, $\bar N=\bar U U_N$, and
$\bar M=\bar U U_M$.

Finally, the moment cotangent returns to the end that governed the maximum. At
that end it is split by the direction cosines $M_y/M$ and $M_z/M$. At zero
moment Normax returns zero. At an exact tie between ends it follows the selected
maximum branch. The landscape is piecewise smooth; the implementation does not
sprinkle differentiability dust over a real branch.

## Crossing the Tesseract boundary

The forward endpoint receives member actions, a held diameter, and the fixed
section constants. It returns demanded diameter and both demanded and held
utilization. The reverse endpoint receives cotangents on those outputs and
applies the rules above to axial force, both end-moment arrays, and held
diameter. Blueprints remains unmodified.

Run the four-way derivative comparison with:

```bash
uv run python validation/blueprint_adjoint.py
```

The script compares the implicit tangent, its reverse transpose, an independent
closed form, and central differences of the host bisection. The derivation is
small. The point is large: code compliance can participate in inverse design
without replacing the code implementation that practitioners recognize.

## What this rule does not cover

This derivative covers axial force with biaxial bending at cross-section level
for the fixed Class 3 CHS family. It does not cover member flexural buckling,
global stability, shear, torsion, self-weight feedback, discrete product
catalogs, connections, fabrication, other section families, or transitions
between section classes. Those omissions are project boundaries, not safety
claims.
