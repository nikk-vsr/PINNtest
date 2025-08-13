# === 3) Bayesian PINN using TensorFlow Probability DenseVariational ===
            # KL losses are in model.losses when using DenseVariational layers
            kl_loss = tf.add_n(bayes_model.losses) if bayes_model.losses else 0.0

            total_loss = data_loss + pde_weight * pde_loss + kl_loss

        grads = tape.gradient(total_loss, bayes_model.trainable_variables + [kappa_var, q_var])
        optimizer.apply_gradients(zip(grads, bayes_model.trainable_variables + [kappa_var, q_var]))

    # logging
    if epoch % 100 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch}: total_loss={total_loss.numpy():.6e}, data_loss={data_loss.numpy():.6e}, pde_loss={pde_loss.numpy():.6e}, kl_loss={kl_loss.numpy() if hasattr(kl_loss,'numpy') else kl_loss:.3e}")

# After training, obtain predictive mean and std using MC sampling
n_samples_mc = 100
X_test_grid = X_all.astype(np.float32)
mc_preds = []
for i in range(n_samples_mc):
    preds = bayes_model(X_test_grid, training=True).numpy().reshape(-1, 1)
    mc_preds.append(preds)
mc_preds = np.stack(mc_preds, axis=0)  # shape (n_samples_mc, n_points, 1)
pred_mean = mc_preds.mean(axis=0)
pred_std = mc_preds.std(axis=0)

print("Bayesian PINN predictive mean/std computed (sampled)")

# Evaluate kappa and q values (point estimates)
print("Bayesian PINN learned kappa (softplus):", tf.nn.softplus(kappa_var).numpy())
print("Bayesian PINN learned q:", q_var.numpy())

# Compute MSE on observed test points
X_test_obs = X_obs_test.astype(np.float32)
idxs = []
# find indices of observed test points in X_test_grid (approx)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1).fit(X_test_grid)
dist, inds = nn.kneighbors(X_test_obs)
obs_preds_mean = pred_mean[inds.flatten()]
mse_obs = np.mean((obs_preds_mean - y_obs_test.reshape(-1, 1)) ** 2)
print("Bayesian PINN observed test MSE:", mse_obs)

# === Plot example: temperature field mean and uncertainty at final time ===
# select final time T_total and plot across x
mask_final = np.isclose(X_all[:, 1], T_total)
if not np.any(mask_final):
    # find closest
    t_vals = X_all[:, 1]
    t_closest = np.max(t_vals)
    mask_final = t_vals == t_closest

x_final = X_all[mask_final][:, 0]
mean_final = pred_mean[mask_final].flatten()
std_final = pred_std[mask_final].flatten()

plt.figure(figsize=(8, 5))
plt.plot(x_final, mean_final, label='Predictive mean')
plt.fill_between(x_final, mean_final - 2 * std_final, mean_final + 2 * std_final, alpha=0.3, label='~95% CI')
plt.title('Bayesian PINN: predictive mean and uncertainty (final time)')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === Save models/artifacts (optional) ===
# model.save('bayes_pinn_model')

# End of script
# -----------------------------------------------------------------
# Notes & Next steps suggestions (in code comments):
# - The Bayesian model above sets priors/posteriors in a simple mean-field way. You can extend priors, apply hierarchical priors
#   or treat PDE parameters (kappa,q) as Bayesian random variables using tfp distributions and variational inference.
# - The PDE residual computation inside the Keras/TF loop is somewhat expensive. You can optimize by vectorized, analytical
#   expressions for second derivatives using higher-order GradientTape patterns.
# - For production/large-scale: consider domain decomposition, transfer learning, and more sophisticated UQ like HMC.
