
import cobra
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

model = cobra.io.read_sbml_model("HepatoNet1.xml")
print("Model loaded:", model.name)

essential_inputs = ['EX_glc(e)', 'EX_hdca(e)', 'EX_cho(e)', 'EX_ocdca(e)']
for rxn_id in essential_inputs:
    if rxn_id in model.reactions:
        model.reactions.get_by_id(rxn_id).lower_bound = -10

tg_pools = ["HC02062_r", "HC02062_c", "HC02062_l"]

def add_tg_demands(m):
    dm_rxns = []
    for met_id in tg_pools:
        if met_id not in m.metabolites:
            continue
        dm_id = f"DM_{met_id}"
        if dm_id in m.reactions:
            dm_rxns.append(m.reactions.get_by_id(dm_id))
        else:
            dm = m.add_boundary(m.metabolites.get_by_id(met_id), type="demand")
            dm.upper_bound = 2000
            dm_rxns.append(dm)
    return dm_rxns

dm_rxns = add_tg_demands(model)
model.objective = {rxn: 1 for rxn in dm_rxns}
model.objective_direction = "max"

baseline_model = deepcopy(model)
nafld_model = deepcopy(model)

tg_clearance_rxns = ["EX_HC02062_r", "r1223", "r1224", "r1264", "r1265"]
#these are reactions the catalyse fat, so we slowed to smuilate a nafld liver
for met_id in tg_pools:
    if met_id in nafld_model.metabolites:
        for rxn in nafld_model.metabolites.get_by_id(met_id).reactions:
            rxn.upper_bound *= 2.0

for rxn_id in tg_clearance_rxns:
    if rxn_id in nafld_model.reactions:
        rxn = nafld_model.reactions.get_by_id(rxn_id)
        rxn.upper_bound *= 0.3
        if rxn.lower_bound < 0:
            rxn.lower_bound *= 0.3

phytochem_model = deepcopy(nafld_model)
toinhibitrxns = ["r1280", "r1281", "r1264", "r1265", "DM_HC02062_r", "DM_HC02062_c", "DM_HC02062_l", "EX_HC02062_r", "r1223", "r1224"]
#these are the reactions that the phytochemicals inhibit, acaca, mpk8, mapk14, hdac3, and dgat2
for rxn_id in toinhibitrxns:
    if rxn_id in phytochem_model.reactions:
        rxn = phytochem_model.reactions.get_by_id(rxn_id)
        rxn.upper_bound *= 0.3

sol_base = baseline_model.optimize()
sol_nafld = nafld_model.optimize()
sol_phyto = phytochem_model.optimize()

def get_fluxes(solution):
    r = solution.fluxes["DM_HC02062_r"]
    c = solution.fluxes["DM_HC02062_c"]
    l = solution.fluxes["DM_HC02062_l"]
    return r, c, l, (r + c + l)

base_vals = list(get_fluxes(sol_base))
nafld_vals = list(get_fluxes(sol_nafld))
phyto_vals = list(get_fluxes(sol_phyto))

err_base = [v * 0.1 for v in base_vals]
err_nafld = [v * 0.1 for v in nafld_vals]
err_phyto = [v * 0.1 for v in phyto_vals]

labels = ["HC02062_r", "HC02062_c", "HC02062_l", "Total TG"]
x = np.arange(len(labels))
width = 0.35

print("\nMetabolic Flux Results")
for name, vals in {"Baseline": base_vals, "NAFLD": nafld_vals, "Phytochem": phyto_vals}.items():
    print(f"\n{name} Scenario:")
    for label, v in zip(labels, vals):
        error = v * 0.10
        print(f"  {label}: {v:.4f} ± {error:.4f}")

plt.figure(figsize=(10, 6))
plt.gcf().patch.set_facecolor("white")
ax1 = plt.gca()
ax1.set_facecolor("white")

ax1.bar(x - width/2, base_vals, width, yerr=err_base, label="Baseline",
        edgecolor="black", capsize=5, error_kw={'ecolor': 'black'})
ax1.bar(x + width/2, nafld_vals, width, yerr=err_nafld, label="NAFLD",
        edgecolor="black", capsize=5, error_kw={'ecolor': 'black'})

ax1.set_ylabel("Flux")
ax1.set_title("TG Accumulation Change (Baseline vs NAFLD)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.tick_params(colors="black")
ax1.legend()

plt.figure(figsize=(10, 6))
plt.gcf().patch.set_facecolor("white")
ax2 = plt.gca()
ax2.set_facecolor("white")

ax2.bar(x - width/2, nafld_vals, width, yerr=err_nafld, label="NAFLD",
        color='#e74c3c', edgecolor="black", capsize=5,
        error_kw={'ecolor': 'black'})
ax2.bar(x + width/2, phyto_vals, width, yerr=err_phyto,
        label="NAFLD + Phytochem", color='#2ecc71',
        edgecolor="black", capsize=5,
        error_kw={'ecolor': 'black'})

ax2.set_ylabel("Flux")
ax2.set_title("Impact of Phytochemical Inhibition")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.tick_params(colors="black")
ax2.legend()

plt.tight_layout()
plt.show()
