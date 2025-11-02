from statsmodels.stats.power import FTestAnovaPower

# Study assumptions
EFFECT_SIZE = 0.25  # Cohen's f
ALPHA = 0.05        # Type I error rate
POWER = 0.80        # Desired power
GROUPS = 6          # Five personas + control

solver = FTestAnovaPower()
# solve_power returns the total sample size for the ANOVA design
total = solver.solve_power(
    effect_size=EFFECT_SIZE,
    alpha=ALPHA,
    power=POWER,
    k_groups=GROUPS,
)

per_group = total / GROUPS

print("Power analysis for one-way ANOVA")
print(f"Total responses needed: {total:.2f}")
print(f"Responses needed per persona: {per_group:.2f}")