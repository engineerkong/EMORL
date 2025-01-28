bandit_arm = Exp3()
bandit = Exp3()
bandit_arm(scaled, chosen)
bandit(1, chosen)
chosen = bandit_arm.draw()
weights = bandit.weights