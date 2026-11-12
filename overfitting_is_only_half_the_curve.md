# Double Descent 101: Your Stats Class Drew Only Half of the Overfitting Curve

## _I kept adding parameters to a least-squares model long after it could fit every noisy training point. Its test error exploded, then fell again and settled 42 percent below the textbook sweet spot_

Every machine learning course draws the same picture on day one. Plot test error against model complexity and you get a U. Too simple and the model underfits, too complex and it overfits, so you pick the bottom of the U and stop there. That picture is the reason we regularize, stop training early and keep models small. **It is correct as far as it goes, and it stops at exactly the wrong place.**

The models that work best today have far more parameters than training examples. They fit their training data perfectly, noise included, and the U says they should be useless. They are not. The explanation has a name, double descent: keep adding capacity past the point where the model has just enough parameters to fit every training point, and test error spikes, then falls a second time.

This usually gets presented as a deep learning curiosity, so I wanted to see it without a neural network or a GPU. I built it from scratch on a 2-core laptop CPU with plain least squares on polynomial and random features, using synthetic data where I know the true function, so test error can be measured against the truth itself. Every number below comes from a notebook or a test in the repository.

[IMAGE: 03_double_descent.png]

Source: author

That is where this article ends up. The left panel fits polynomials to 15 noisy points, the right panel fits random features to 100 points in 20 dimensions. Both curves have the same three parts: the familiar U, a spike where the number of parameters equals the number of training points, and a second descent. On the right, the model with 2,000 parameters, twenty for every data point, reproduces every training label and still has a lower test error than the best model the U would have picked.

## Two models and a known truth

Both models are linear regressions on a set of features, so capacity is simply the number of features p, and the fit is ordinary least squares. With p smaller than the number of training points n, least squares has one answer. With p larger than n, infinitely many coefficient vectors fit the training data perfectly, and the pseudo-inverse picks the one with the smallest length. Nearly everything below depends on that rule, and it fits in a few lines of NumPy:

```python
def min_norm_fit(phi, y):
    u, s, vt = np.linalg.svd(phi, full_matrices=False)
    keep = s > np.finfo(float).eps * max(phi.shape) * s[0]
    return vt.T[:, keep] @ ((u.T[keep] @ y) / s[keep])
```

The first model is a polynomial in one variable. The true function is sin(pi x) + 0.5 cos(2 pi x) on the interval from -1 to 1, with Gaussian noise of standard deviation 0.3 added to 15 training points. I used Legendre polynomials instead of the usual x, x squared, x cubed. Raw powers of x make the least-squares problem numerically meaningless somewhere around degree 15 to 20, and round-off error can then fake or hide the very effect I am trying to measure.

The second model works on 20-dimensional inputs. The target is tanh(2 times the projection of x on a fixed random direction), and each feature is max(0, w.x + b) with w and b drawn at random and then frozen. Only the final linear layer is fitted, so this is a two-layer neural network whose first layer never trains. It is the simplest model that can be made as wide as you like.

Test error is always the mean squared error against the noiseless true function on 2,000 fixed test points, and every curve is the median over many independent training sets (200 for the polynomial, 40 for the random features), with the interquartile range as a band.

## The textbook U is real

Start where every course starts, on the left side of p = n.

[IMAGE: 01_classical_u.png]

Source: author

The polynomial's median test error falls from 0.65 with a single parameter to 0.12 with five, then climbs by orders of magnitude as p approaches 15. The random features show the same shape in 20 dimensions, with a sweet spot at 32 features and an error of 0.42. Training error, the dashed line, only ever goes down.

The usual explanation holds up when you measure it. Refitting the random-feature model on 600 independent training sets and splitting the expected error into squared bias and variance, the bias falls from 0.62 to 0.11 across the underparameterized range while the variance rises from 0.02 to 2.3. Their sum is the U.

So far, nothing to argue with. Most courses stop drawing here, somewhere to the left of p = n.

## The worst model is exactly as big as the data

Keep the sweep going and the error climbs into a sharp spike.

[IMAGE: 02_peak_at_threshold.png]

Source: author

With any noise in the labels, the maximum of the median curve lands exactly on p = n: degree 14 for 15 points, 100 features for 100 points. For the polynomial at noise 0.3 the median test error there is about 3 billion, ten orders of magnitude above the sweet spot. The random features peak at 71, about 170 times their best classical error. The peak also tracks the noise. For random features it grows from 30 to 81 as the noise goes from 0 to 0.5, and the noiseless polynomial has no peak at all.

[IMAGE: 02_contorted_fit.png]

Source: author

This is what the worst model looks like. With as many parameters as points there is one and only one polynomial that passes through all 15 of them, noise included. Its training error is 3e-23, and between the points it swings out to about 110,000.

The reason it happens at p = n and nowhere else is linear algebra. The fitted coefficients are roughly the labels divided by the smallest singular value of the feature matrix, and a square random matrix is almost always close to singular. For the random features the median smallest singular value is 0.67 at half the threshold, 0.012 at the threshold and 0.95 at twice the threshold. **The peak is label noise amplified by a nearly singular matrix.** Below the threshold, least squares cannot fit the noise and averages it away. Above it, the extra columns move the matrix away from singular again.

## Then the error comes back down

Past the threshold the training error is zero to machine precision, around 1e-24, and stays there. The test error falls. For the random features it keeps falling all the way to 0.24 at 2,000 features, 42 percent below the classical sweet spot. The polynomial falls by ten orders of magnitude to 0.16 at 90 parameters, which is far better than anything near the threshold but still 28 percent above its own sweet spot. I come back to that difference below.

[IMAGE: 03_bias_variance_across_threshold.png]

Source: author

The decomposition shows where the second descent comes from. Squared bias drops in the classical regime and then sits flat at about 0.10 on both sides of the threshold, apart from a blip at p = n itself. The variance does all the moving. It spikes to an average of 5,370 at the threshold, where the fit depends entirely on the particular noise draw, then falls to 0.14 at 2,000 features. The classical decomposition still holds. What fails is the assumption that variance only ever grows with capacity.

## My first two attempts went wrong in instructive ways

The first version of this project used random features on one-dimensional inputs, because then I could draw every fitted curve. The second descent never came. Past the threshold the test error kept growing with width, and the peak was not even at p = n. In one dimension, a smooth curve forced through noisy points has nowhere to put the noise except into wiggles between neighbouring points, so overfitting there is at best what Mallinar and colleagues (2022) call tempered rather than benign. The same code on 20-dimensional inputs produced the textbook curve on the first run. That is also why the 1D polynomial levels off above its sweet spot while the 20D model beats its own.

The second surprise came from averages. I first estimated the polynomial's bias and variance with plain means, and the expected error exploded at just six parameters, long before the threshold. One training set in a few thousand happens to contain two almost identical x values, and that single draw dominates the mean of all the others. The median curve was perfectly clean. **Near the threshold, report medians, or say clearly why you report means.**

## Which interpolant you get is the whole story

Past p = n every model in the sweep has zero training error, so fitting the data perfectly cannot be what separates good models from bad ones. What separates them is which of the infinitely many perfect fits the solver returns.

[IMAGE: 03_which_interpolant.png]

Source: author

"Smallest coefficients" depends on how the features are scaled. In the polynomial runs I scale the k-th Legendre polynomial down by 1/(1+k), which makes high-order wiggles expensive, so the minimum-length fit prefers smooth explanations. Remove that scaling and every degree costs the same. Below the threshold nothing changes, because least squares has a unique answer there. Past the threshold the unscaled fit spends its budget on narrow spikes at the training points and drops back towards zero in between, and its median error at 90 parameters is 0.54, close to the 0.63 of a model that always predicts zero. The scaled fit gets 0.16 on the same data, with the same zero training error.

## Zero training error is not doom

The uncomfortable part deserves a closer look: a model that memorises noisy labels and still predicts well.

[IMAGE: 04_memorize_and_generalize.png]

Source: author

On one training set, the classical sweet spot (32 features) has a training error of 0.22 and a test error of 0.40. The 2,000-feature model reproduces every noisy label, with a training error of 4e-30, and its test error is 0.25. Its predictions on new points sit visibly tighter around the true function.

Because least squares is linear in the labels, the fit splits cleanly into a part fitted to the clean signal and a part fitted to the noise. On the training points the noise part reproduces the noise exactly. On test points its contribution to the error is 27 at the threshold and 0.064 at 2,000 features, smaller than the noise variance of 0.09 that it memorised. With 2,000 random directions in 20 dimensions, most of them barely matter for predictions at new points, and that is where the noise ends up. This is the intuition behind benign overfitting (Bartlett and colleagues, 2020). Across noise levels 0, 0.1, 0.3 and 0.5, the interpolating model beats the classical sweet spot by 51, 51, 42 and 27 percent.

## Adding data can make the model worse

The threshold is about the ratio of parameters to data, so you can also reach it from the other side. I fixed the random-feature model at 100 features and grew the training set from 10 to 400 points, each set containing the previous one.

[IMAGE: 05_more_data_hurts.png]

Source: author

At noise 0.3 the best training-set size below 100 is about 30 points, with an error of 0.48. Every point added after that makes the model worse, up to 53 at 100 points, more than a hundred times higher. Only past n = p does more data help again, reaching 0.21 at 400 points. Even with noiseless labels the error grows 81 times on its way to n = p, because the part of the target that 100 random features cannot represent acts like noise. Nakkiran and colleagues (2020) found the same sample-wise effect in deep networks. More data is not unconditionally better for a model sized close to its data.

## A little ridge removes the peak

Everything so far used the ridgeless fit on purpose, to show the effect in its rawest form. Ridge regression adds a penalty on the size of the coefficients, which is the lever the threshold model was abusing.

[IMAGE: 06_ridge_reshapes_curve.png]

Source: author

A penalty of 0.0001 per training point cuts the error at the threshold from 67 to 2.8. At 0.01 the peak is gone. At 0.1 the curve matches or beats the ridgeless one at every size, and at 1.0 it reaches the lowest error of any random-feature model in this study, 0.16 against a ridgeless floor of 0.25, at the cost of being up to 13 percent worse for small models. Nakkiran and colleagues (2021) showed that well-tuned regularization can remove double descent in settings like this one. Regularization is not obsolete. It decides whether the threshold is a cliff or a gentle slope.

Random features freeze the first layer, so I also trained a one-hidden-layer ReLU network with both layers learning, full-batch Adam and no weight decay. Its median test error is 0.12 with one hidden unit, peaks at 0.96 with five units (111 parameters, the first width at which the training error reaches zero), and falls to 0.24 with 136 units. The peak sits near p = n rather than exactly on it, because a trained network's effective capacity depends on optimisation. On this target the wide network does not beat the single unit, which already matches a function of one direction almost perfectly. The shape carries over to trained networks; whether the far side wins depends on the problem.

## Where the U still holds

[IMAGE: 06_regime_map.png]

Source: author

The U is not wrong, it covers one regime. For models with fewer parameters than training examples, cross-validated model size, early stopping and regularization behave as taught. Near p = n, expect erratic test error that looks like bad luck and is in fact a predictable peak, and do not expect small changes in data or capacity to help in a straight line. Far past the threshold, zero training error is not a warning sign on its own and more capacity often helps, especially in high dimensions, but the features, architecture and optimiser decide which interpolant you get, and a tuned penalty usually still improves it.

## What I would take away from this

- Compute the ratio of parameters (or effective capacity) to training examples before reasoning about overfitting.
- The worst model is the one with as many parameters as data points, not the biggest one.
- Past the threshold, test error can fall again with training error at zero, and in high dimensions it can beat the classical sweet spot.
- Whether interpolation is benign depends on which interpolant you get, and that is set by feature scaling, dimension and the fitting method.
- Near the threshold, more data can hurt, averages mislead, and a small ridge penalty is the cheapest fix.

The code, notebooks, figures and a small Streamlit app are on GitHub at https://github.com/Dima806/double_descent_101. Run `make setup` and `make notebooks`, and every number in this article reproduces on a 2-core machine in under eight minutes. For the pieces around it, regularization_101 (https://github.com/Dima806/regularization_101) builds the penalties that flatten the peak, curse_of_dimensionality_101 (https://github.com/Dima806/curse_of_dimensionality_101) explores the high-dimensional geometry where benign overfitting lives, and cv_arena (https://github.com/Dima806/cv_arena) looks at how we estimate the test error that behaves so strangely near the threshold.

## References

1. M. Belkin, D. Hsu, S. Ma, S. Mandal, Reconciling Modern Machine-Learning Practice and the Classical Bias-Variance Trade-off (2019), PNAS
2. P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, I. Sutskever, Deep Double Descent: Where Bigger Models and More Data Hurt (2020), ICLR
3. S. Geman, E. Bienenstock, R. Doursat, Neural Networks and the Bias/Variance Dilemma (1992), Neural Computation
4. C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals, Understanding Deep Learning Requires Rethinking Generalization (2017), ICLR
5. P. L. Bartlett, P. M. Long, G. Lugosi, A. Tsigler, Benign Overfitting in Linear Regression (2020), PNAS
6. T. Hastie, A. Montanari, S. Rosset, R. J. Tibshirani, Surprises in High-Dimensional Ridgeless Least Squares Interpolation (2022), Annals of Statistics
7. P. Nakkiran, P. Venkat, S. Kakade, T. Ma, Optimal Regularization Can Mitigate Double Descent (2021), ICLR
8. N. Mallinar, J. B. Simon, A. Abedsoltan, P. Pandit, M. Belkin, P. Nakkiran, Benign, Tempered, or Catastrophic: Toward a Refined Taxonomy of Overfitting (2022), NeurIPS
