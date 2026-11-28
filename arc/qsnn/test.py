net.eval()
with torch.no_grad():
    hidden_spikes, output_spikes = net(X.to(device))
    all_rates = output_rates(output_spikes).cpu().numpy()
    all_logits = RATE_LOGIT_SCALE * all_rates
    all_probabilities = torch.softmax(torch.as_tensor(all_logits), dim=1).numpy()
    all_predictions = all_logits.argmax(axis=1)
    output_spike_counts = output_spikes.sum(dim=0).cpu().numpy()
    total_hidden_spikes = hidden_spikes.sum(dim=(0,2)).cpu().numpy()

scores_class1 = all_probabilities[:,1]
test_pred = all_predictions[test_idx]
test_true = labels[test_idx]
metrics = {
    "accuracy": accuracy_score(test_true,test_pred),
    "precision_class1": precision_score(test_true,test_pred,zero_division=0),
    "recall_class1": recall_score(test_true,test_pred,zero_division=0),
    "f1_class1": f1_score(test_true,test_pred,zero_division=0),
}
conf = confusion_matrix(test_true,test_pred,labels=[0,1])
monotonic_violations = int(np.sum(np.diff(scores_class1[np.argsort(h_values)]) < -1e-6))
silent_fraction = float(np.mean(total_hidden_spikes == 0))

print("--- Test reservado ---")
for key,value in metrics.items(): print(f"{key:18s}: {value:.4f}")
print("Matriz de confusión [real × predicha]:\\n",conf)
print(f"Descensos locales del score: {monotonic_violations}/{len(scores_class1)-1}")
print(f"Ejemplos sin spikes ocultos: {silent_fraction:.3f}")

sort_idx = np.argsort(h_values/J)
fig,ax=plt.subplots(figsize=(7.4,4.4))
ax.plot(h_values[sort_idx]/J,scores_class1[sort_idx],marker="o",markersize=2.6,linewidth=1.25,color="black",label="QSNN: tasa de salida + softmax")
ax.axvline(H_C/J,color="0.4",linestyle="--",linewidth=1,label=fr"$h_c/J={H_C/J:.1f}$")
ax.axhline(0.5,color="0.65",linestyle=":",linewidth=1)
ax.set(xlabel=r"Campo transversal $h/J$",ylabel="Probabilidad de fase 1",xlim=(H_MIN/J,H_MAX/J),ylim=(-0.03,1.03))
ax.grid(alpha=0.2); ax.legend(frameon=False); fig.tight_layout(); plt.show()

fig,ax=plt.subplots(figsize=(7.4,4.4))
ax.scatter(h_values[dev_idx]/J,scores_class1[dev_idx],s=20,label="Desarrollo")
ax.scatter(h_values[test_idx]/J,scores_class1[test_idx],s=26,marker=".",label="Test")
ax.axvline(H_C/J,color="0.4",linestyle="--",linewidth=1)
ax.set(xlabel=r"Campo transversal $h/J$",ylabel="Probabilidad de fase 1",ylim=(-0.03,1.03))
ax.grid(alpha=0.2); ax.legend(frameon=False); fig.tight_layout(); plt.show()

results_df = pd.DataFrame({
    "h": h_values, "h_over_J": h_values/J, "label": labels,
    "predicted_label": all_predictions, "score_class_0": all_probabilities[:,0],
    "score_class_1": scores_class1, "rate_phase_0": all_rates[:,0],
    "rate_phase_1": all_rates[:,1], "spikes_phase_0": output_spike_counts[:,0],
    "spikes_phase_1": output_spike_counts[:,1], "total_hidden_spikes": total_hidden_spikes,
    "energy_vqe": energies, "energy_exact": exact_energies,
    "energy_error": energy_errors, "is_test": np.isin(np.arange(len(h_values)), test_idx)
}).sort_values("h").reset_index(drop=True)
results_df.to_csv(RESULTS_DIR / "qsnn_scores_dual_alpha.csv", index=False)
with open(RESULTS_DIR / "qsnn_metrics_dual_alpha.json", "w", encoding="utf-8") as f:
    json.dump({**metrics, "best_epoch": int(best_epoch), "train_time_s": float(train_time),
               "hidden_neurons": int(NUM_HIDDEN), "output_neurons": 2, "num_steps": int(NUM_STEPS),
               "num_parameters": int(num_parameters), "test_size": int(len(test_idx)),
               "validation_size": int(len(val_idx)), "monotonic_violations": monotonic_violations,
               "rate_logit_scale": RATE_LOGIT_SCALE, "loss": "cross_entropy_on_output_firing_rates"}, f, indent=2)
torch.save(net.state_dict(), RESULTS_DIR / "qsnn_dual_alpha_state_dict.pt")
print(f"Resultados guardados en {RESULTS_DIR.resolve()}")
